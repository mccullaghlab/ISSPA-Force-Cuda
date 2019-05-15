
#include "angle_class.h"

// Allocate angle arrays on host (cpu)
void angle::allocate()
{
	//
	//angleAtoms_h= (int *)malloc(3*nAngles*sizeof(int));
	cudaMallocHost((int **) &angleAtoms_h, 3*nAngles*sizeof(int));
	cudaMallocHost((int **) &angleKs_h, nAngles*sizeof(float));
	cudaMallocHost((int **) &angleX0s_h, nAngles*sizeof(float));
	//angleKs_h= (float *)malloc(nAngles*sizeof(float));
	//angleX0s_h= (float *)malloc(nAngles*sizeof(float));
	angleKUnique = (float *)malloc(nTypes*sizeof(float));
	angleX0Unique = (float *)malloc(nTypes*sizeof(float));
}
// Allocate angle arrays on device (gpu) and send data
void angle::initialize_gpu()
{
	//
	cudaMalloc((void **) &angleX0s_d, nAngles*sizeof(float));
	cudaMalloc((void **) &angleKs_d, nAngles*sizeof(float));
	cudaMalloc((void **) &angleAtoms_d, nAngles*3*sizeof(int));
	// copy data to device
	cudaMemcpy(angleAtoms_d, angleAtoms_h, nAngles*3*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(angleKs_d, angleKs_h, nAngles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(angleX0s_d, angleX0s_h, nAngles*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventCreate(&angleStart);
	cudaEventCreate(&angleStop);

}


void angle::free_arrays() {
	// free host variables
	cudaFree(angleAtoms_h);
	cudaFree(angleKs_h);
	cudaFree(angleX0s_h);
	free(angleKUnique);
	free(angleX0Unique);
}

void angle::free_arrays_gpu() {
	// free device variables
	cudaFree(angleX0s_d);
	cudaFree(angleKs_d);
	cudaFree(angleAtoms_d);
}
