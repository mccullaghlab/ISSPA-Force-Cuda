
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include "constants.h"
#include "bond_class.h"

// Allocate bond arrays on host (cpu)
void bond::allocate()
{
	//
	//bondAtoms_h= (int *)malloc(2*nBonds*sizeof(int));
	cudaMallocHost((int**) &bondAtoms_h, 2*nBonds*sizeof(int));
	cudaMallocHost((float**) &bondKs_h, nBonds*sizeof(float));
	cudaMallocHost((float**) &bondX0s_h, nBonds*sizeof(float));
	//bondKs_h= (float *)malloc(nBonds*sizeof(float));
	//bondX0s_h= (float *)malloc(nBonds*sizeof(float));
	bondKUnique = (float *)malloc(nTypes*sizeof(float));
	bondX0Unique = (float *)malloc(nTypes*sizeof(float));
}
// Allocate bond arrays on device (gpu) and send data
void bond::initialize_gpu()
{
	//
	cudaMalloc((void **) &bondX0s_d, nBonds*sizeof(float));
	cudaMalloc((void **) &bondKs_d, nBonds*sizeof(float));
	cudaMalloc((void **) &bondAtoms_d, nBonds*2*sizeof(int));
	// copy data to device
	cudaMemcpy(bondAtoms_d, bondAtoms_h, nBonds*2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(bondKs_d, bondKs_h, nBonds*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bondX0s_d, bondX0s_h, nBonds*sizeof(float), cudaMemcpyHostToDevice);

	// get gridSize and blockSize
	bond_force_cuda_grid_block(nBonds, &gridSize, &blockSize, &minGridSize);

}


void bond::free_arrays() {
	// free host variables
	cudaFree(bondAtoms_h);
	cudaFree(bondKs_h);
	cudaFree(bondX0s_h);
	free(bondKUnique);
	free(bondX0Unique);
}

void bond::free_arrays_gpu() {
	// free device variables
	cudaFree(bondX0s_d);
	cudaFree(bondKs_d);
	cudaFree(bondAtoms_d);
}
