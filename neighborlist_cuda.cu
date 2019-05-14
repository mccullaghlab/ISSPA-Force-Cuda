
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "atom_class.h"
#include "neighborlist_cuda.h"

#define nDim 3

// CUDA Kernels

__global__ void neighborlist_kernel(float *xyz, int *NN, int *numNN, float rNN2, int nAtoms, int numNNmax, float lbox, int *nExcludedAtoms, int *excludedAtomsList, int excludedAtomsListLength) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x ;
	extern __shared__ int excludedAtomsList_s[];
	int atom1;
	int atom2;
	float temp, dist2;	
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
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) {
					temp = __ldg(xyz+atom1*nDim+k) - __ldg(xyz+atom2*nDim+k);
					//temp = xyz_s[atom1*nDim+k] - xyz_s[atom2*nDim+k];
					if (temp > hbox) {
						temp -= lbox;
					} else if (temp < -hbox) {
						temp += lbox;
					}
					dist2 += temp*temp;
				}
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

//extern "C" float neighborlist_cuda(float *xyz_d, int *NN_d, int *numNN_d, float rNN2, int nAtoms, int numNNmax, float lbox, int *nExcludedAtoms_d, int *excludedAtomsList_d, int excludedAtomsListLength) {
float neighborlist_cuda(atom& atoms, float rNN2, float lbox)
{
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 
	cudaEvent_t neighborListStart, neighborListStop;
	float milliseconds;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, neighborlist_kernel, 0, atoms.nAtoms); 

    	// Round up according to array size 
    	gridSize = (atoms.nAtoms + blockSize - 1) / blockSize; 
	// initialize cuda timing events
	cudaEventCreate(&neighborListStart);
	cudaEventCreate(&neighborListStop);
	cudaEventRecord(neighborListStart);
	// run nonbond cuda kernel
	neighborlist_kernel<<<gridSize, blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.xyz_d, atoms.NN_d, atoms.numNN_d, rNN2, atoms.nAtoms, atoms.numNNmax, lbox, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.excludedAtomsListLength);
	// record kernel timing
	cudaEventRecord(neighborListStop);
    	cudaEventSynchronize(neighborListStop);
	cudaEventElapsedTime(&milliseconds, neighborListStart, neighborListStop);
	return milliseconds;

}

