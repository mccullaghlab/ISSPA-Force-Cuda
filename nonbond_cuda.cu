
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

	if (index < nAtoms)
	{
		hbox = lbox/2.0;
		// determine two atoms to work on based on recursive definition
		atom1 = index;
		f[atom1*nDim] = f[atom1*nDim+1] = f[atom1*nDim+2] = 0.0f;
		for (atom2=0;atom2<nAtoms;atom2++) {
			if (atom2 != atom1) {
				// get interaction type
				it = ityp[atom1];
				jt = ityp[atom2];
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) {
					r[k] = xyz[atom1*nDim+k] - xyz[atom2*nDim+k];
					if (r[k] > hbox) {
//						r[k] -= (int)(temp/lbox+0.5) * lbox;
						r[k] -= lbox;
					} else if (r[k] < -hbox) {
//						r[k] += (int)(-temp/lbox+0.5) * lbox;
						r[k] += lbox;
					}
					dist2 += r[k]*r[k];
				}
				// LJ force
				r2 = 1.0 / dist2;
				r6 = r2 * r2 * r2;
				fs = r6 * (12.0 * lj_A[it] * r6 - 6.0 * lj_B[it]) / dist2;
				f[atom1*nDim] += fs*r[0];
				f[atom1*nDim+1] += fs*r[1];
				f[atom1*nDim+2] += fs*r[2];
//				atomicAdd(&f[atom2*nDim], -fs*r[0] );
//				atomicAdd(&f[atom2*nDim+1], -fs*r[1] );
//				atomicAdd(&f[atom2*nDim+2], -fs*r[2] );
			}
		}

	}
}

/* C wrappers for kernels */

extern "C" void nonbond_cuda(float *xyz_d, float *f_d, float *charges_d, float *lj_A_d, float *lj_B_d, int *ityp_d, int nAtoms, float lbox) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nonbond_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	gridSize = (nAtoms + blockSize - 1) / blockSize; 

	// run nonbond cuda kernel
	nonbond_kernel<<<gridSize, blockSize>>>(xyz_d, f_d, charges_d, lj_A_d, lj_B_d, ityp_d, nAtoms, lbox);

}

