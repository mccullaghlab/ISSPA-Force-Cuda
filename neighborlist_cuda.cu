
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "neighborlist_cuda.h"

#define nDim 3

// CUDA Kernels

__global__ void neighborlist_kernel(float4 *xyz, int *NN, int *numNN, float rNN2, int nAtoms, int numNNmax, float lbox, int *nExcludedAtoms, int *excludedAtomsList, int excludedAtomsListLength) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x ;
	extern __shared__ int excludedAtomsList_s[];
	int atom1;
	int atom2;
	float4 temp;
	float dist2;	
	int i, k;
	int count;
	int start;
	int exStart;
	int exStop;
	int exPass;
	int exAtom;
	int chunk;
	float hbox;

	// copy excluded atoms list to shared memory
	chunk = (int) ( (excludedAtomsListLength+blockDim.x-1)/blockDim.x);
	for (i=t*chunk;i<(t+1)*chunk;i++) {
		excludedAtomsList_s[i] = excludedAtomsList[i];
	}
	__syncthreads();
	// move on

	if (index < nAtoms)
	{
		hbox = lbox/2.0;
		atom1 = index;
		start = atom1*numNNmax;
		count = 0;
		for (atom2=0;atom2<nAtoms;atom2++) {
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
					if (excludedAtomsList_s[exAtom]-1 == atom2) {
						exPass = 1;
						break;
					}
					// the following only applies if exluded atom list is in strictly ascending order
					if (excludedAtomsList_s[exAtom]-1 > atom2) {
						break;
					}
				}
			} else if (atom1 > atom2) {
				if (atom2==0) {
					exStart = 0;
				} else {
					//exStart = nExcludedAtoms[atom2-1];
					exStart = __ldg(nExcludedAtoms+atom2-1);
				}
				exStop = __ldg(nExcludedAtoms+atom2);
				for (exAtom=exStart;exAtom<exStop;exAtom++) {
					if (excludedAtomsList_s[exAtom]-1 == atom1) {
						exPass = 1;
						break;
					}
					// the following only applies if exluded atom list is in strictly ascending order
					if (excludedAtomsList_s[exAtom]-1 > atom1) {
						break;
					}
				}
			}
			if (atom2 != atom1 && exPass == 0) {
				// compute distance
				temp = min_image(__ldg(xyz+atom1) - __ldg(xyz+atom2),lbox,hbox);
				dist2 = temp.x*temp.x + temp.y*temp.y + temp.z*temp.z;
				if (dist2 < rNN2) {
					NN[start+count] = atom2;
					count ++;
				}
			}
		}
		numNN[atom1] = count;
	}
}

/* C wrappers for kernels */

float neighborlist_cuda(atom& atoms, float rNN2, float lbox)
{
	float milliseconds;

	// initialize cuda timing events
	cudaEventRecord(atoms.neighborListStart);
	// run nonbond cuda kernel
	neighborlist_kernel<<<atoms.gridSize, atoms.blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.pos_d, atoms.NN_d, atoms.numNN_d, rNN2, atoms.nAtoms, atoms.numNNmax, lbox, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.excludedAtomsListLength);
	// record kernel timing
	cudaEventRecord(atoms.neighborListStop);
    	cudaEventSynchronize(atoms.neighborListStop);
	cudaEventElapsedTime(&milliseconds, atoms.neighborListStart, atoms.neighborListStop);
	return milliseconds;

}

