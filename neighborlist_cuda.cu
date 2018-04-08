
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "neighborlist_cuda.h"

#define nDim 3
//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// CUDA Kernels

__global__ void neighborlist_kernel(float *xyz, int *NN, int *numNN, float rNN2, int nAtoms, int numNNmax, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom1;
	int atom2;
	float temp, dist2;	
	int i, k;
	int count;
	int start;
	float hbox;

	if (index < nAtoms)
	{
		hbox = lbox/2.0;
		// determine two atoms to work on based on recursive definition
		atom1 = index;
		start = atom1*numNNmax;
		count = 0;
		for (atom2=0;atom2<nAtoms;atom2++) {
			if (atom2 != atom1) {
				// compute distance
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) {
					temp = xyz[atom1*nDim+k] - xyz[atom2*nDim+k];
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

extern "C" void neighborlist_cuda(float *xyz_d, int *NN_d, int *numNN_d, float rNN2, int nAtoms, int numNNmax, float lbox) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, neighborlist_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	gridSize = (nAtoms + blockSize - 1) / blockSize; 

	// run nonbond cuda kernel
	neighborlist_kernel<<<gridSize, blockSize>>>(xyz_d, NN_d, numNN_d, rNN2, nAtoms, numNNmax, lbox);

}

