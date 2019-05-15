
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
	cudaMallocHost((int**) &bondAtoms_h, nBonds*sizeof(int2));
	cudaMallocHost((float**) &bondKs_h, nBonds*sizeof(float));
	cudaMallocHost((float**) &bondX0s_h, nBonds*sizeof(float));
	bondKUnique = (float *)malloc(nTypes*sizeof(float));
	bondX0Unique = (float *)malloc(nTypes*sizeof(float));
}
// Allocate bond arrays on device (gpu) and send data
void bond::initialize_gpu()
{
	//
	cudaMalloc((void **) &bondX0s_d, nBonds*sizeof(float));
	cudaMalloc((void **) &bondKs_d, nBonds*sizeof(float));
	cudaMalloc((void **) &bondAtoms_d, nBonds*sizeof(int2));
	// copy data to device
	cudaMemcpy(bondAtoms_d, bondAtoms_h, nBonds*sizeof(int2), cudaMemcpyHostToDevice);
	cudaMemcpy(bondKs_d, bondKs_h, nBonds*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bondX0s_d, bondX0s_h, nBonds*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventCreate(&bondStart);
	cudaEventCreate(&bondStop);

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
