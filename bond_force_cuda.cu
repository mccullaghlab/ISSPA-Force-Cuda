
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "constants.h"
#include "bond_class.h"
#include "bond_force_cuda.h"

// Texture reference for 2D float texture
//texture<float, 1, cudaReadModeElementType> tex;

// CUDA Kernels

__global__ void bond_force_kernel(float *xyz, float *f, int nAtoms, float lbox, int *bondAtoms, float *bondKs, float *bondX0s, int nBonds) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	//extern __shared__ float xyz_s[];
	int atom1;
	int atom2;
	float dist2;	
	int k;
	float r[nDim];
	float fbnd;
	float hbox;
	
	//if (t < nAtoms*nDim) {
	//	xyz_s[t] = xyz[t];	
//		__syncthreads();
//	}

	if (index < nBonds)
	{
		hbox = lbox/2.0;
		// determine two atoms to work  - these will be unique to each index
		atom1 = __ldg(bondAtoms+index*2);
		atom2 = __ldg(bondAtoms+index*2+1);
		dist2 = 0.0f;
		for (k=0;k<nDim;k++) {
//			r[k] = __ldg(xyz+atom1+k) - __ldg(xyz+atom2+k);
			r[k] = xyz[atom1+k] - xyz[atom2+k];
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

//extern "C" float bond_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, int *bondAtoms_d, float *bondKs_d, float *bondX0s_d, int nBonds, int gridSize, int blockSize) 
float bond_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, bond& bonds) 
{
	cudaEvent_t bondStart, bondStop;
	float milliseconds;
	// Set texture parameters
	//tex.addressMode[0] = cudaAddressModeWrap;
	//tex.filterMode = cudaFilterModeLinear;
	//tex.normalized = true;    // access with normalized texture coordinates

	// Bind the array to the texture
	//cudaBindTexture(0, tex, xyz_d, nAtoms*nDim*sizeof(float));
	// initialize cuda kernel timing
	cudaEventCreate(&bondStart);
	cudaEventCreate(&bondStop);
	cudaEventRecord(bondStart);
	// run nonbond cuda kernel
	//bond_force_kernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(xyz_d, f_d, nAtoms, lbox, bondAtoms_d, bondKs_d, bondX0s_d, nBonds);
	bond_force_kernel<<<bonds.gridSize, bonds.blockSize>>>(xyz_d, f_d, nAtoms, lbox, bonds.bondAtoms_d, bonds.bondKs_d, bonds.bondX0s_d, bonds.nBonds);
	cudaEventRecord(bondStop);
	cudaEventSynchronize(bondStop);
	cudaEventElapsedTime(&milliseconds, bondStart, bondStop);
	return milliseconds;

}

extern "C" void bond_force_cuda_grid_block(int nBonds, int *gridSize, int *blockSize, int *minGridSize)
{

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, bond_force_kernel, 0, nBonds); 
    	// Round up according to array size 
    	*gridSize = (nBonds + *blockSize - 1) / *blockSize; 

}
