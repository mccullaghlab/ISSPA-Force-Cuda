
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector_functions.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "nonbond_force_cuda.h"

#define nDim 3

// CUDA Kernels

__global__ void nonbond_force_kernel(float4 *xyz, float4 *f, float2 *lj, int nAtoms, float rCut2, float lbox, int *nExcludedAtoms, int *excludedAtomsList, int excludedAtomsListLength, int *nbparm, int *ityp, int nPairs, int nTypes) {
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
			hbox = lbox/2.0;
			p1 = __ldg(xyz + atom1);
			p2 = __ldg(xyz + atom2);
			r = min_image(p1 - p2,lbox,hbox);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			if (dist2 < rCut2) {
				// LJ pair type
				it = __ldg(ityp+atom1);
				jt = __ldg(ityp+atom2);
				nlj = nTypes*(it)+jt;
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

float nonbond_force_cuda(atom &atoms, float rCut2, float lbox) 
{
	int gridSize;
	int blockSize;
	int minGridSize;
	float milliseconds;

	// timing
	cudaEventRecord(atoms.nonbondStart);
	
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nonbond_force_kernel, 0, atoms.nPairs); 
    	// Round up according to array size 
    	gridSize = (atoms.nPairs + blockSize - 1) / blockSize; 
	// run nonbond cuda kernel
	nonbond_force_kernel<<<gridSize, blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.pos_d, atoms.for_d, atoms.lj_d, atoms.nAtoms, rCut2, lbox, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.excludedAtomsListLength, atoms.nonBondedParmIndex_d, atoms.ityp_d, atoms.nPairs, atoms.nTypes);

	// finish timing
	cudaEventRecord(atoms.nonbondStop);
	cudaEventSynchronize(atoms.nonbondStop);
	cudaEventElapsedTime(&milliseconds, atoms.nonbondStart, atoms.nonbondStop);
	return milliseconds;

}

extern "C" void nonbond_force_cuda_grid_block(int nAtoms, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, nonbond_force_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	*gridSize = (nAtoms + *blockSize - 1) / *blockSize; 

}
