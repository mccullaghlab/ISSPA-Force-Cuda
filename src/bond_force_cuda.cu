
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_vector_routines.h"
#include "constants.h"
#include "bond_class.h"
#include "bond_force_cuda.h"

// Texture reference for 2D float texture
//texture<float, 1, cudaReadModeElementType> tex;

// CUDA Kernels

__global__ void bond_force_kernel(float4 *xyz, float4 *f, int nAtoms, float lbox, int4 *bondAtoms, float2 *bondParams, int nBonds, int nBondTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	//unsigned int t = threadIdx.x;
	//extern __shared__ float2 bondParams_s[];
	int4 atoms;
	float dist;	
	float4 r;
	float2 params;
	float fbnd;
	float hbox;

	// store bond params on shared memory	
	//if (t < nBondTypes) {
	//	bondParams_s[t] = __ldg(bondParams+t);	
	//}
	//__syncthreads();
	// force calculation for each bond
	if (index < nBonds)
	{
		hbox = 0.5f*lbox;
		// determine two atoms to work  third element of int4 contains bond type
		atoms = __ldg(bondAtoms+index);
		params = __ldg(bondParams+atoms.z);
		// get distance vector separating the two atoms
		r = min_image(__ldg(xyz+atoms.x) - __ldg(xyz+atoms.y),lbox,hbox);
		// compute reciprocal of distance
		dist = rnorm3df(r.x,r.y,r.z);
		// force
		fbnd = params.x*(params.y*dist - 1.0f);
		// add force to force vector
		atomicAdd(&(f[atoms.x].x), fbnd*r.x);
		atomicAdd(&(f[atoms.y].x), -fbnd*r.x);
		atomicAdd(&(f[atoms.x].y), fbnd*r.y);
		atomicAdd(&(f[atoms.y].y), -fbnd*r.y);
		atomicAdd(&(f[atoms.x].z), fbnd*r.z);
		atomicAdd(&(f[atoms.y].z), -fbnd*r.z);

	}
}

/* C wrappers for kernels */

//extern "C" float bond_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, int *bondAtoms_d, float *bondKs_d, float *bondX0s_d, int nBonds, int gridSize, int blockSize) 
float bond_force_cuda(float4 *xyz_d, float4 *f_d, int nAtoms, float lbox, bond& bonds) 
{
	float milliseconds;
	// Set texture parameters
	//tex.addressMode[0] = cudaAddressModeWrap;
	//tex.filterMode = cudaFilterModeLinear;
	//tex.normalized = true;    // access with normalized texture coordinates

	// Bind the array to the texture
	//cudaBindTexture(0, tex, xyz_d, nAtoms*nDim*sizeof(float));
	// initialize cuda kernel timing
	cudaEventRecord(bonds.bondStart);
	// run nonbond cuda kernel
	//bond_force_kernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(xyz_d, f_d, nAtoms, lbox, bondAtoms_d, bondKs_d, bondX0s_d, nBonds);
	//printf("bonds.gridSize= %d and bonds.blockSize=%d\n",bonds.gridSize,bonds.blockSize);
	bond_force_kernel<<<bonds.gridSize, bonds.blockSize >>>(xyz_d, f_d, nAtoms, lbox, bonds.bondAtoms_d, bonds.bondParams_d, bonds.nBonds, bonds.nTypes);
	cudaEventRecord(bonds.bondStop);
	cudaEventSynchronize(bonds.bondStop);
	cudaEventElapsedTime(&milliseconds, bonds.bondStart, bonds.bondStop);
	return milliseconds;

}

extern "C" void bond_force_cuda_grid_block(int nBonds, int *gridSize, int *blockSize, int *minGridSize)
{

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, bond_force_kernel, 0, nBonds); 
    	// Round up according to array size 
    	*gridSize = (nBonds + *blockSize - 1) / *blockSize; 

}
