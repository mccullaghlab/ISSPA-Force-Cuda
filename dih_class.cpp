
#include "dih_class.h"

// Allocate dih arrays on host (cpu)
void dih::allocate()
{
	//
	//dihAtoms_h= (int *)malloc(4*nDihs*sizeof(int));
	//dihKs_h= (float *)malloc(nDihs*sizeof(float));
	//dihNs_h= (float *)malloc(nDihs*sizeof(float));
	//dihPs_h= (float *)malloc(nDihs*sizeof(float));
	cudaMallocHost((int **) &dihAtoms_h, 5*nDihs*sizeof(int));
	cudaMallocHost((float **) &dihKs_h, nTypes*sizeof(float));
	cudaMallocHost((float **) &dihNs_h, nTypes*sizeof(float));
	cudaMallocHost((float **) &dihPs_h, nTypes*sizeof(float));
//	dihKUnique = (float *)malloc(nTypes*sizeof(float));
//	dihNUnique = (float *)malloc(nTypes*sizeof(float));
//	dihPUnique = (float *)malloc(nTypes*sizeof(float));
	// non-bonded scale factors
	cudaMallocHost((float **) &sceeScaleFactor_h, nTypes*sizeof(float));
	cudaMallocHost((float **) &scnbScaleFactor_h, nTypes*sizeof(float));
}
// Allocate dih arrays on device (gpu) and send data
void dih::initialize_gpu()
{
	//
	cudaMalloc((void **) &dihNs_d, nTypes*sizeof(float));
	cudaMalloc((void **) &dihPs_d, nTypes*sizeof(float));
	cudaMalloc((void **) &dihKs_d, nTypes*sizeof(float));
	cudaMalloc((void **) &dihAtoms_d, nDihs*5*sizeof(int));
	cudaMalloc((void **) &sceeScaleFactor_d, nTypes*sizeof(float));
	cudaMalloc((void **) &scnbScaleFactor_d, nTypes*sizeof(float));
	// copy data to device
	cudaMemcpy(dihAtoms_d, dihAtoms_h, nDihs*5*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dihKs_d, dihKs_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dihNs_d, dihNs_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dihPs_d, dihPs_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sceeScaleFactor_d, sceeScaleFactor_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(scnbScaleFactor_d, scnbScaleFactor_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);

}


void dih::free_arrays() {
	// free host variables
	cudaFree(dihAtoms_h);
	cudaFree(dihKs_h);
	cudaFree(dihNs_h);
	cudaFree(dihPs_h);
	cudaFree(sceeScaleFactor_h);
	cudaFree(scnbScaleFactor_h);
}

void dih::free_arrays_gpu() {
	// free device variables
	cudaFree(dihNs_d);
	cudaFree(dihPs_d);
	cudaFree(dihKs_d);
	cudaFree(dihAtoms_d);
	cudaFree(sceeScaleFactor_d);
	cudaFree(scnbScaleFactor_d);
}
