
#include "dih_class.h"

// Allocate dih arrays on host (cpu)
void dih::allocate()
{
	//
	dihAtoms_h= (int *)malloc(4*nDihs*sizeof(int));
	dihKs_h= (float *)malloc(nDihs*sizeof(float));
	dihNs_h= (float *)malloc(nDihs*sizeof(float));
	dihPs_h= (float *)malloc(nDihs*sizeof(float));
	dihKUnique = (float *)malloc(nTypes*sizeof(float));
	dihNUnique = (float *)malloc(nTypes*sizeof(float));
	dihPUnique = (float *)malloc(nTypes*sizeof(float));
}
// Allocate dih arrays on device (gpu) and send data
void dih::initialize_gpu()
{
	//
	cudaMalloc((void **) &dihNs_d, nDihs*sizeof(float));
	cudaMalloc((void **) &dihPs_d, nDihs*sizeof(float));
	cudaMalloc((void **) &dihKs_d, nDihs*sizeof(float));
	cudaMalloc((void **) &dihAtoms_d, nDihs*4*sizeof(int));
	// copy data to device
	cudaMemcpy(dihAtoms_d, dihAtoms_h, nDihs*4*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dihKs_d, dihKs_h, nDihs*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dihNs_d, dihNs_h, nDihs*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dihPs_d, dihPs_h, nDihs*sizeof(float), cudaMemcpyHostToDevice);

}


void dih::free_arrays() {
	// free host variables
	free(dihAtoms_h);
	free(dihKs_h);
	free(dihNs_h);
	free(dihPs_h);
	free(dihKUnique);
	free(dihNUnique);
	free(dihPUnique);
}

void dih::free_arrays_gpu() {
	// free device variables
	cudaFree(dihNs_d);
	cudaFree(dihPs_d);
	cudaFree(dihKs_d);
	cudaFree(dihAtoms_d);
}
