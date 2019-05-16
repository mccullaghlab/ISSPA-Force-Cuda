
#include "angle_class.h"

// Allocate angle arrays on host (cpu)
void angle::allocate()
{
	//angleAtoms_h= (int *)malloc(3*nAngles*sizeof(int));
	cudaMallocHost((int **) &angleAtoms_h, nAngles*sizeof(int4));
	cudaMallocHost((int **) &angleParams_h, nTypes*sizeof(float2));
}
// Allocate angle arrays on device (gpu) and send data
void angle::initialize_gpu()
{
	int i;
	//
	cudaMalloc((void **) &angleAtoms_d, nAngles*sizeof(int4));
	cudaMalloc((void **) &angleParams_d, nTypes*sizeof(float2));
	// copy data to device
	cudaMemcpy(angleAtoms_d, angleAtoms_h, nAngles*sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(angleParams_d, angleParams_h, nTypes*sizeof(float2), cudaMemcpyHostToDevice);
	cudaEventCreate(&angleStart);
	cudaEventCreate(&angleStop);

}


void angle::free_arrays() {
	// free host variables
	cudaFree(angleAtoms_h);
	cudaFree(angleParams_h);
}

void angle::free_arrays_gpu() {
	// free device variables
	cudaFree(angleParams_d);
	cudaFree(angleAtoms_d);
}
