
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "bond_force_cuda.h"
#include "constants.h"

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// CUDA Kernels

__global__ void bond_force_kernel(float *xyz, float *f, int nAtoms, float lbox, int *bondAtoms, float *bondKs, float *bondX0s, int nBonds) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float xyz_s[];
	extern __shared__ int bondAtoms_s[];
	int atom1;
	int atom2;
	float dist2;	
	int k;
	float r[nDim];
	float fbnd;
	float hbox;
	
	if (t < nAtoms*nDim) {
		xyz_s[t] = xyz[t];	
		__syncthreads();
	}

	if (index < nBonds)
	{
		hbox = lbox/2.0;
		// determine two atoms to work  - these will be unique to each index
		atom1 = bondAtoms[index*2];
		atom2 = bondAtoms[index*2+1];
		dist2 = 0.0f;
		for (k=0;k<nDim;k++) {
			r[k] = xyz_s[atom1+k] - xyz_s[atom2+k];
			// assuming no more than one box away
			if (r[k] > hbox) {
				r[k] -= lbox;
			} else if (r[k] < -hbox) {
				r[k] += lbox;
			}
			dist2 += r[k]*r[k];
		}
		fbnd = bondKs[index]*(bondX0s[index]/sqrtf(dist2) - 1.0f);
		for (k=0;k<3;k++) {
			//temp = fbnd*r[k];
			atomicAdd(&f[atom1+k], fbnd*r[k]);
			atomicAdd(&f[atom2+k], -fbnd*r[k]);
		}

	}
}

/* C wrappers for kernels */

extern "C" void bond_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, int *bondAtoms_d, float *bondKs_d, float *bondX0s_d, int nBonds) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bond_force_kernel, 0, nBonds); 

    	// Round up according to array size 
    	gridSize = (nBonds + blockSize - 1) / blockSize; 
	blockSize = nAtoms*nDim;
	gridSize = 1;
	// run nonbond cuda kernel
	bond_force_kernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(xyz_d, f_d, nAtoms, lbox, bondAtoms_d, bondKs_d, bondX0s_d, nBonds);

}

