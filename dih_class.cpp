
#include "dih_class.h"

// Allocate dih arrays on host (cpu)
void dih::allocate()
{
	//
	cudaMallocHost((int4 **) &dihAtoms_h, nDihs*sizeof(int4));
	cudaMallocHost((int **) &dihTypes_h, nDihs*sizeof(int));
	cudaMallocHost((float **) &dihParams_h, nTypes*sizeof(float4));
	// non-bonded scale factors
	cudaMallocHost((float **) &sceeScaleFactor_h, nTypes*sizeof(float));
	cudaMallocHost((float **) &scnbScaleFactor_h, nTypes*sizeof(float));
	//
	cudaEventCreate(&dihStart);
	cudaEventCreate(&dihStop);
}
// Allocate dih arrays on device (gpu) and send data
void dih::initialize_gpu()
{
	//
	cudaMalloc((void **) &dihParams_d, nTypes*sizeof(float4));
	cudaMalloc((void **) &dihAtoms_d, nDihs*sizeof(int4));
	cudaMalloc((void **) &dihTypes_d, nDihs*sizeof(int));
	cudaMalloc((void **) &sceeScaleFactor_d, nTypes*sizeof(float));
	cudaMalloc((void **) &scnbScaleFactor_d, nTypes*sizeof(float));
	// copy data to device
	cudaMemcpy(dihAtoms_d, dihAtoms_h, nDihs*sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(dihTypes_d, dihTypes_h, nDihs*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dihParams_d, dihParams_h, nTypes*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(sceeScaleFactor_d, sceeScaleFactor_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(scnbScaleFactor_d, scnbScaleFactor_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);

}


void dih::free_arrays() {
	// free host variables
	cudaFree(dihAtoms_h);
	cudaFree(dihTypes_h);
	cudaFree(dihParams_h);
	cudaFree(sceeScaleFactor_h);
	cudaFree(scnbScaleFactor_h);
}

void dih::free_arrays_gpu() {
	// free device variables
	cudaFree(dihParams_d);
	cudaFree(dihAtoms_d);
	cudaFree(dihTypes_d);
	cudaFree(sceeScaleFactor_d);
	cudaFree(scnbScaleFactor_d);
}
