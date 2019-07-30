
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector_functions.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "nonbond_force_cuda.h"

#define nDim 3

// constants
__constant__ float rCut2;
__constant__ float lbox;
__constant__ float hbox;
__constant__ int nAtoms;
__constant__ int nPairs;
__constant__ int nTypes;
__constant__ int excludedAtomsListLength;

// CUDA Kernels

__global__ void nonbond_force_kernel(float4 *xyz, float4 *f, float2 *lj, int *nExcludedAtoms, int *excludedAtomsList, int *nbparm, int *ityp) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ int excludedAtomsList_s[];
	int atom1, atom2;
	float dist2;	
	int i;
	int exStart;
	int exStop;
	int exPass;
	int exAtom;
	int it, jt;    // atom type of atom of interest
	int nlj;
	float4 r;
	float r6;
	float fc;
	float flj;
	float2 ljAB;
	float4 p1, p2;
	int nPairs_l = nPairs;
	int nAtoms_l = nAtoms;
	float hbox_l = hbox;
	float lbox_l = lbox;
	int nTypes_l = nTypes;
	float rCut2_l = rCut2;



	// copy excluded atoms list to shared memory
	for (i=t;i<excludedAtomsListLength;i+=blockDim.x) {
		excludedAtomsList_s[i] = __ldg(excludedAtomsList+i);
	}
	__syncthreads();
	// move on

	if (index < nPairs_l)
	{
		atom1 = (int) (index/nAtoms_l);
		atom2 = index % nAtoms_l;
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
		if (atom1 != atom2 && exPass == 0) {
			p1 = __ldg(xyz + atom1);
			p2 = __ldg(xyz + atom2);
			r = min_image(p1 - p2,lbox_l,hbox_l);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			if (dist2 < rCut2_l) {
				// LJ pair type
				it = __ldg(ityp+atom1);
				jt = __ldg(ityp+atom2);
				nlj = nTypes_l*(it)+jt;
				nlj = __ldg(nbparm+nlj);
				ljAB = __ldg(lj+nlj);
				// LJ force
				r6 = powf(dist2,-3.0);
				flj = r6 * (12.0 * ljAB.x * r6 - 6.0 * ljAB.y) / dist2;
				// coulomb force
				fc = p1.w*p2.w/dist2/sqrtf(dist2);
				// add forces to atom1
				atomicAdd(&(f[atom1].x),(flj+fc)*r.x);
				atomicAdd(&(f[atom1].y),(flj+fc)*r.y);
				atomicAdd(&(f[atom1].z),(flj+fc)*r.z);
				// add forces to atom2
				//atomicAdd(&(f[atoms.y].x),-(flj+fc)*r.x);
				//atomicAdd(&(f[atoms.y].y),-(flj+fc)*r.y);
				//atomicAdd(&(f[atoms.y].z),-(flj+fc)*r.z);

			}
		}

	}
}

/* C wrappers for kernels */

float nonbond_force_cuda(atom &atoms)
{
	float milliseconds;

	// timing
	cudaEventRecord(atoms.nonbondStart);
	
	// run nonbond cuda kernel
	nonbond_force_kernel<<<atoms.gridSize, atoms.blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.pos_d, atoms.for_d, atoms.lj_d, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.nonBondedParmIndex_d, atoms.ityp_d);

	// finish timing
	cudaEventRecord(atoms.nonbondStop);
	cudaEventSynchronize(atoms.nonbondStop);
	cudaEventElapsedTime(&milliseconds, atoms.nonbondStart, atoms.nonbondStop);
	return milliseconds;

}

extern "C" void nonbond_force_cuda_grid_block(atom& atoms, float rCut2_h, float lbox_h)
{
	int minGridSize;
	float hbox_h = lbox_h/2;

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
	cudaMemcpyToSymbol(hbox, &hbox_h, sizeof(float));
	
}
