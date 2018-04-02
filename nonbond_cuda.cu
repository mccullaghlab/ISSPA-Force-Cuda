
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "nonbond_cuda.h"

#define nDim 3
//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

//__global__ void init_rand(unsigned int long seed, curandState_t* states){
//	curand_init(seed,blockIdx.x,0,&states);
//}
// CUDA Kernels

__global__ void nonbond_kernel(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom1;
	int atom2;
	int it;    // atom type of atom of interest
	int jt;    // atom type of other atom
	float temp, dist2;	
	int i, k;
	int count;
	float r[3];
	float r2, r6, fs;
	float hbox;

	if (index < nAtoms*(nAtoms-1)/2)
	{
		hbox = lbox/2.0;
		// determine two atoms to work on based on recursive definition
		count = 0;
		for (i=0;i<nAtoms-1;i++) {
			count += nAtoms-1-i;
			if (index < count) {
				atom1 = i;	
				atom2 = nAtoms - count + index;
				break;
			}
		}
		// get interaction type
		it = ityp[atom1];
		jt = ityp[atom2];
		dist2 = 0.0f;
		for (k=0;k<nDim;k++) {
			r[k] = xyz[atom1*nDim+k] - xyz[atom2*nDim+k];
			if (r[k] > hbox) {
				r[k] -= (int)(temp/hbox) * lbox;
			} else if (r[k] < -hbox) {
				r[k] += (int)(temp/hbox) * lbox;
			}
			dist2 += r[k]*r[k];
		}
		// LJ force
		r2 = 1/dist2;
		r6 = r2 * r2 * r2;
		fs = r6 * (lj_B[it] - lj_A[it] * r6);
		atomicAdd(&f[atom1*nDim], fs*r[0] );
		atomicAdd(&f[atom1*nDim+1], fs*r[1] );
		atomicAdd(&f[atom1*nDim+2], fs*r[2] );
		atomicAdd(&f[atom2*nDim], -fs*r[0] );
		atomicAdd(&f[atom2*nDim+1], -fs*r[1] );
		atomicAdd(&f[atom2*nDim+2], -fs*r[2] );

	}
}

/* C wrappers for kernels */

extern "C" void nonbond_cuda(float *xyz_d, float *f_d, float *charges_d, float *lj_A_d, float *lj_B_d, int *ityp_d, int nAtoms, float lbox) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nonbond_kernel, 0, nAtoms*(nAtoms-1)/2); 

    	// Round up according to array size 
    	gridSize = (nAtoms*(nAtoms-1)/2 + blockSize - 1) / blockSize; 

	// run nonbond cuda kernel
	nonbond_kernel<<<gridSize, blockSize>>>(xyz_d, f_d, charges_d, lj_A_d, lj_B_d, ityp_d, nAtoms, lbox);

}

