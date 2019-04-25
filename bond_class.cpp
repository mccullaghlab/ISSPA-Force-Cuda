
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
	bondAtoms_h= (int *)malloc(2*nBonds*sizeof(int));
	bondKs_h= (float *)malloc(nBonds*sizeof(float));
	bondX0s_h= (float *)malloc(nBonds*sizeof(float));
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

}
