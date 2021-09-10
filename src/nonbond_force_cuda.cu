#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector_functions.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "isspa_class.h"
#include "nonbond_force_cuda.h"

#define nDim 3

// constants
__constant__ float rCut2;
__constant__ float lbox;
__constant__ int nAtoms;
__constant__ int nPairs;
__constant__ int nTypes;
__constant__ int excludedAtomsListLength;

// CUDA Kernels
__inline__ __device__
float4 warpReduceSumTriple(float4 val) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val.x += __shfl_down(val.x, offset);
                val.y += __shfl_down(val.y, offset);
                val.z += __shfl_down(val.z, offset);
        }
        return val; 
}

__global__ void nonbond_force_kernel(float4 *xyz, float4 *f, float4 *isspaf, float2 *lj, const float* __restrict__ rmax, int *isspaTypes, int *nExcludedAtoms, int *excludedAtomsList, int *nbparm, int *ityp, int nThreads) {
        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int tid = threadIdx.x;
	extern __shared__ int excludedAtomsList_s[];
	int atom1, atom2;
	float dist, dist2;	
	int i;
	int exStart;
	int exStop;
	int exPass;
	int exAtom;
	int it, jt;    // atom type of atom of interest
	int nlj;
	float4 r;
	float rmax_l;
	float r6;
	float fc;
	float flj;
	float fdir;
	float hbox;
	float2 ljAB;
	float4 p1, p2;
        

        
	// copy excluded atoms list to shared memory
	for (i=tid;i<excludedAtomsListLength;i+=blockDim.x) {
		excludedAtomsList_s[i] = __ldg(excludedAtomsList+i);
	}
	__syncthreads();
        
	// Determine atom indices
        atom1 = int(double(index)/double(nThreads));
        atom2 = int(index - atom1*nThreads);

	// all threads need to set r to zero
	r.x = r.y = r.z = r.w = 0.0f;
	if (atom2 < nAtoms)
	{

		// check exclusions
		exPass = 0;
		if (atom1 < atom2) {
			if (atom1==0) {
				exStart = 0;
			} else {
				//exStart = nExcludedAtoms[atom1-1];
				exStart = __ldg(nExcludedAtoms+atom1-1);
			}
			exStop = __ldg(nExcludedAtoms+atom1);
			for (exAtom=exStart;exAtom<exStop;exAtom++) {
				if (excludedAtomsList_s[exAtom] == atom2) {
					exPass = 1;
					break;
				}
				// the following only applies if exluded atom list is in strictly ascending order
				if (excludedAtomsList_s[exAtom] > atom2) {
					break;
				}
			}
		} else if (atom2 < atom1) {
			if (atom2==0) {
				exStart = 0;
			} else {
				exStart = __ldg(nExcludedAtoms+atom2-1);
			}
			exStop = __ldg(nExcludedAtoms+atom2);
			for (exAtom=exStart;exAtom<exStop;exAtom++) {
				if (excludedAtomsList_s[exAtom] == atom1) {
					exPass = 1;
					break;
				}
				// the following only applies if exluded atom list is in strictly ascending order
				if (excludedAtomsList_s[exAtom] > atom1) {
					break;
				}
			}
		}
		// finish exclusion check
		if (atom1 != atom2) {
			hbox = 0.5f*lbox;
			// load atom data
			p1 = __ldg(xyz + atom1);
			p2 = __ldg(xyz + atom2);
			// get IS-SPA rmax
			jt = __ldg(isspaTypes + atom2);
			rmax_l = rmax[jt];
			// compute separation vector and distance between them
			r = min_image(p1 - p2,lbox,hbox);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			dist = sqrtf(dist2);
			// if not in excluded list
		        if (exPass == 0) {
			       // LJ pair type
			       it = __ldg(ityp+atom1);
			       jt = __ldg(ityp+atom2);
			       nlj = nTypes*(it)+jt;
			       nlj = __ldg(nbparm+nlj);
			       ljAB = __ldg(lj+nlj);
			       // LJ force
			       r6 = powf(dist2,-3.0f);
			       flj = __fdividef(r6 * (12.0f * ljAB.x * r6 - 6.0f * ljAB.y),dist2);
			       //atomicAdd(&(isspaf[atom1].x),(flj)*r.x);
			       //atomicAdd(&(isspaf[atom1].y),(flj)*r.y);
			       //atomicAdd(&(isspaf[atom1].z),(flj)*r.z);
			       fc = __fdividef(p1.w*p2.w,dist2*dist);
			       //atomicAdd(&(isspaf[atom1].x),(fc)*r.x);
			       //atomicAdd(&(isspaf[atom1].y),(fc)*r.y);
			       //atomicAdd(&(isspaf[atom1].z),(fc)*r.z);
			       // IS-SPA long ranged electrostatics
			       if (dist > 2.0f*rmax_l) {
				       // coulomb force
				       fdir = __fdividef(-2.0f*p1.w*p2.w*(1.0f-__fdividef(1.0f,ep)),3.0f*dist2*dist);
			       } else {
				       //fdir = -p1.w*p2.w*(1.0f-1.0f/ep)*(8.0f*rmax_l-3.0f*dist)/24.0f/(rmax_l*rmax_l*rmax_l*rmax_l);
				       fdir = __fdividef(-p1.w*p2.w*(1.0f-__fdividef(1.0f,ep))*(8.0f*rmax_l-3.0f*dist),24.0f*__powf(rmax_l,4.0f));
			       }
			       //add forces
			       //atomicAdd(&(isspaf[atom1].x),(fdir)*r.x);
			       //atomicAdd(&(isspaf[atom1].y),(fdir)*r.y);
			       //atomicAdd(&(isspaf[atom1].z),(fdir)*r.z);
			       fdir += flj + fc;
			} else {
			       	// IS-SPA long ranged electrostatics
				if (dist > 2.0f*rmax_l) {
				       //fdir = -p1.w*p2.w/dist2/dist*2.0f/3.0f*(1.0f-1.0f/ep);
				       fdir = __fdividef(-2.0f*p1.w*p2.w*(1.0f-__fdividef(1.0f,ep)),3.0f*dist2*dist);
			       	} else {
				       //fdir = -p1.w*p2.w*(1.0f-1.0f/ep)*(8.0f*rmax_l-3.0f*dist)/24.0f/(rmax_l*rmax_l*rmax_l*rmax_l);
				       fdir = __fdividef(-p1.w*p2.w*(1.0f-__fdividef(1.0f,ep))*(8.0f*rmax_l-3.0f*dist),24.0f*__powf(rmax_l,4.0f));
			       	}
                                //atomicAdd(&(isspaf[atom1].x),(fdir)*r.x);
                                //atomicAdd(&(isspaf[atom1].y),(fdir)*r.y);
                                //atomicAdd(&(isspaf[atom1].z),(fdir)*r.z);			       
                        }
			// finalize force vector
			r *= fdir;
		}
	} 
	// warp reduce the force
	r = warpReduceSumTriple(r);
	if((tid & (warpSize - 1)) == 0) {
		atomicAdd(&(f[atom1].x),r.x);
		atomicAdd(&(f[atom1].y),r.y);
		atomicAdd(&(f[atom1].z),r.z);
	}
}

/* C wrappers for kernels */

float nonbond_force_cuda(atom& atoms, isspa& isspas, int nAtoms_h)
{
	float milliseconds;

	// timing
	cudaEventRecord(atoms.nonbondStart);
	
	// run nonbond cuda kernel
	nonbond_force_kernel<<<atoms.gridSize, atoms.blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.pos_d, atoms.for_d, atoms.isspaf_d, atoms.lj_d, isspas.rmax_d, isspas.isspaTypes_d, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.nonBondedParmIndex_d, atoms.ityp_d, atoms.nThreads);
	
	// finish timing
	cudaEventRecord(atoms.nonbondStop);
	cudaEventSynchronize(atoms.nonbondStop);
	cudaEventElapsedTime(&milliseconds, atoms.nonbondStart, atoms.nonbondStop);
	return milliseconds;

}

extern "C" void nonbond_force_cuda_grid_block(atom& atoms, float rCut2_h, float lbox_h)
{
        int maxThreadsPerBlock = 1024;
        
	// determine gridSize and blockSize for nonbond kernel	
        atoms.nThreads = int(ceil( (atoms.nAtoms) / (float) 32.0))*32;
	atoms.blockSize = maxThreadsPerBlock;
        atoms.gridSize = int(ceil( (atoms.nThreads*atoms.nAtoms) / (float) maxThreadsPerBlock));        
                
	printf("Nonbond kernel gridSize = %d\n", atoms.gridSize);
	printf("Nonbond kernel blockSize = %d\n", atoms.blockSize);

	// determine gridSize and blockSize
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &atoms.blockSize, nonbond_force_kernel, 0, atoms.nPairs);

    	// Round up according to array size 
    	//atoms.gridSize = (atoms.nPairs + atoms.blockSize - 1) / atoms.blockSize; 

	// set constants
	cudaMemcpyToSymbol(nAtoms, &atoms.nAtoms, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &atoms.nPairs, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &atoms.nTypes, sizeof(int));
	cudaMemcpyToSymbol(excludedAtomsListLength, &atoms.excludedAtomsListLength, sizeof(int));
	cudaMemcpyToSymbol(rCut2, &rCut2_h, sizeof(float));
	cudaMemcpyToSymbol(lbox, &lbox_h, sizeof(float));
	
}
