
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cell_list_cuda.h"

#define nDim 3
//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// CUDA Kernels

__global__ void cell_list_kernel(float *xyz, int *cellList, int *cellCount, float rCut, int nAtoms, int *nCells, int cellMax, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom;
	int cell;
	float temp, dist2;	
	int k;
	float hbox;

	if (index < nAtoms)
	{
		hbox = lbox/2.0;
		// atom for each thread
		atom = index;
		// determine cell in xyz
		cell = int(xyz[atom*nDim]/rCut)*nCells[1]*nCells[2];
		cell += int(xyz[atom*nDim+1]/rCut)*nCells[2];
		cell += int(xyz[atom*nDim+2]/rCut);
		// add to cells
		atomicAdd(&cellList[cell*cellMax+cellCount[cell]],atom);
		atomicAdd(&cellCount[cell], 1);

	}
}

/* C wrappers for kernels */

extern "C" void cell_list_cuda(float *xyz_d, int *cellList_d, int *cellCount_d, float rCut, int nAtoms, int *nCells, int cellMax, float lbox) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// zero cell arrays
	cudaMemset(cellList_d, 0,  nCells[0]*nCells[1]*nCells[2]*cellMax*sizeof(int));
	cudaMemset(cellCount_d, 0,  nCells[0]*nCells[1]*nCells[2]*sizeof(int));

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cell_list_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	gridSize = (nAtoms + blockSize - 1) / blockSize; 

	// run nonbond cuda kernel
	cell_list_kernel<<<gridSize, blockSize>>>(xyz_d, cellList_d, cellCount_d, rCut, nAtoms, cellMax, lbox);

}

