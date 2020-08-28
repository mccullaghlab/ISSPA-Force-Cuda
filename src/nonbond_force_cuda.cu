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

__global__ void nonbond_force_kernel(float4 *xyz, float4 *f, float4 *isspaf, float2 *lj, float *rmax, int *isspaTypes, int *nExcludedAtoms, int *excludedAtomsList, int *nbparm, int *ityp) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
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
	for (i=t;i<excludedAtomsListLength;i+=blockDim.x) {
		excludedAtomsList_s[i] = __ldg(excludedAtomsList+i);
	}
	__syncthreads();
	// move on

	if (index < nPairs)
	{
		atom1 = (int) (index/nAtoms);
		atom2 = index % nAtoms;
		it = __ldg(isspaTypes + atom2);
		rmax_l = __ldg(rmax+it);
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
		  hbox = lbox/2.0;
		        if (exPass == 0) {
			  
			       p1 = __ldg(xyz + atom1);
			       p2 = __ldg(xyz + atom2);
			       r = min_image(p1 - p2,lbox,hbox);
			       dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			       dist = sqrtf(dist2);
			  
			       // LJ pair type
			       it = __ldg(ityp+atom1);
			       jt = __ldg(ityp+atom2);
			       nlj = nTypes*(it)+jt;
			       nlj = __ldg(nbparm+nlj);
			       ljAB = __ldg(lj+nlj);
			       // LJ force
			       r6 = powf(dist2,-3.0);
			       flj = r6 * (12.0 * ljAB.x * r6 - 6.0 * ljAB.y) / dist2;
			       //atomicAdd(&(isspaf[atom1].x),(flj)*r.x);
			       //atomicAdd(&(isspaf[atom1].y),(flj)*r.y);
			       //atomicAdd(&(isspaf[atom1].z),(flj)*r.z);
			       fc = p1.w*p2.w/dist2/sqrtf(dist2);
			       //atomicAdd(&(isspaf[atom1].x),(fc)*r.x);
			       //atomicAdd(&(isspaf[atom1].y),(fc)*r.y);
			       //atomicAdd(&(isspaf[atom1].z),(fc)*r.z);
			       if (dist > 2.0*rmax_l) {
				       // coulomb force
				       fdir = -p1.w*p2.w/dist2/dist*2.0/3.0*(1.0-1.0/ep);
				       //fc += p1.w*p2.w/dist2/dist*(1.0+2.0/ep)/3.0;
			       } else {
				       fdir = -p1.w*p2.w*(1.0-1.0/ep)*(8.0*rmax_l-3.0*dist)/24.0/(rmax_l*rmax_l*rmax_l*rmax_l);
				       //fc += p1.w*p2.w/dist2/dist*(1.0-1.0/ep)*(8.0*rmax_l-3.0*dist)/24.0/(rmax_l*rmax_l*rmax_l*rmax_l);
			       }
			       // add forces to atom1
			       atomicAdd(&(f[atom1].x),(flj+fc+fdir)*r.x);
			       atomicAdd(&(f[atom1].y),(flj+fc+fdir)*r.y);
			       atomicAdd(&(f[atom1].z),(flj+fc+fdir)*r.z);
			       //atomicAdd(&(isspaf[atom1].x),(fdir)*r.x);
			       //atomicAdd(&(isspaf[atom1].y),(fdir)*r.y);
			       //atomicAdd(&(isspaf[atom1].z),(fdir)*r.z);
			} else {
			       
			        p1 = __ldg(xyz + atom1);
				p2 = __ldg(xyz + atom2);
				r = min_image(p1 - p2,lbox,hbox);
				dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
				dist = sqrtf(dist2);
				if (dist > 2.0*rmax_l) {
				        fdir = -p1.w*p2.w/dist2/dist*2.0/3.0*(1.0-1.0/ep);
					//fc = 0;
				} else {
				        fdir = -p1.w*p2.w*(1.0-1.0/ep)*(8.0*rmax_l-3.0*dist)/24.0/(rmax_l*rmax_l*rmax_l*rmax_l);
					//fc = 0;
				}
				// add forces to atom1
				atomicAdd(&(f[atom1].x),fdir*r.x);
				atomicAdd(&(f[atom1].y),fdir*r.y);
				atomicAdd(&(f[atom1].z),fdir*r.z);
			 	//atomicAdd(&(isspaf[atom1].x),(fdir)*r.x);
				//atomicAdd(&(isspaf[atom1].y),(fdir)*r.y);
				//atomicAdd(&(isspaf[atom1].z),(fdir)*r.z);
			}
		}
	}
}

/* C wrappers for kernels */

float nonbond_force_cuda(atom& atoms, isspa& isspas, int nAtoms_h)
{
	float milliseconds;
	//float4 out_h[nAtoms_h*nAtoms_h]; 

	// timing
	cudaEventRecord(atoms.nonbondStart);
	
	// run nonbond cuda kernel
	nonbond_force_kernel<<<atoms.gridSize, atoms.blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.pos_d, atoms.for_d, atoms.isspaf_d, atoms.lj_d, isspas.rmax_d, isspas.isspaTypes_d, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.nonBondedParmIndex_d, atoms.ityp_d);

	// DEBUG
	//udaMemcpy(out_h, isspas.out_d, nAtoms_h*nAtoms_h*sizeof(float4), cudaMemcpyDeviceToHost);
	//or (int i=0;i<=nAtoms_h*nAtoms_h; i++)
	 // {
	 //   printf("  %15.10f  %15.10f  %15.10f  %15.10f\n", out_h[i].x, out_h[i].y, out_h[i].z, out_h[i].w);
	 // } 
	
	// finish timing
	cudaEventRecord(atoms.nonbondStop);
	cudaEventSynchronize(atoms.nonbondStop);
	cudaEventElapsedTime(&milliseconds, atoms.nonbondStart, atoms.nonbondStop);
	return milliseconds;

}

extern "C" void nonbond_force_cuda_grid_block(atom& atoms, float rCut2_h, float lbox_h)
{
	int minGridSize;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &atoms.blockSize, nonbond_force_kernel, 0, atoms.nPairs);

    	// Round up according to array size 
    	atoms.gridSize = (atoms.nPairs + atoms.blockSize - 1) / atoms.blockSize; 

	// set constants
	cudaMemcpyToSymbol(nAtoms, &atoms.nAtoms, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &atoms.nPairs, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &atoms.nTypes, sizeof(int));
	cudaMemcpyToSymbol(excludedAtomsListLength, &atoms.excludedAtomsListLength, sizeof(int));
	cudaMemcpyToSymbol(rCut2, &rCut2_h, sizeof(float));
	cudaMemcpyToSymbol(lbox, &lbox_h, sizeof(float));
	
}
