
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector_functions.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "nonbond_force_cuda.h"

#define nDim 3

// CUDA Kernels

__global__ void nonbond_force_kernel(float4 *xyz, float4 *f, float2 *lj, int nAtoms, float rCut2, float lbox, int4 *neighborList, int *neighborCount, int nTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float2 lj_s[];
	int4 atoms;
	float dist2;	
	int i, k;
	int N;
	int start;
	int typePairs = nTypes*(nTypes+1)/2;
	float4 r;
	float r6;
	float fc;
	float flj;
	float hbox;
	float2 ljAB;
	float4 p1, p2;
	int nlj;


	// copy lj parameters from global memory to shared memory for each block
	for (i=t;i<typePairs;i+=blockDim.x) {
		lj_s[i] = __ldg(lj+i);
	}
	__syncthreads();
	// move on
	if (index < neighborCount[0])
	{
		hbox = lbox/2.0;
		atoms = __ldg(neighborList+index);
		p1 = __ldg(xyz + atoms.x);
		p2 = __ldg(xyz + atoms.y);
		r = min_image(p1-p2,lbox,hbox);
		dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
		if (dist2 < rCut2) {
			// LJ force
			r6 = powf(dist2,-3.0);
			flj = r6 * (12.0 * lj_s[atoms.z].x * r6 - 6.0 * lj_s[atoms.z].y) / dist2;
			//ljAB = __ldg(lj+atoms.z);
			//flj = r6 * (12.0 * ljAB.x * r6 - 6.0 * ljAB.y) / dist2;
			// coulomb force
			fc = p1.w*p2.w/dist2/sqrtf(dist2);
			// add forces to atom1
			atomicAdd(&(f[atoms.x].x),(flj+fc)*r.x);
			atomicAdd(&(f[atoms.x].y),(flj+fc)*r.y);
			atomicAdd(&(f[atoms.x].z),(flj+fc)*r.z);
			// add forces to atom2
			atomicAdd(&(f[atoms.y].x),-(flj+fc)*r.x);
			atomicAdd(&(f[atoms.y].y),-(flj+fc)*r.y);
			atomicAdd(&(f[atoms.y].z),-(flj+fc)*r.z);

		}

	}
}

/* C wrappers for kernels */

float nonbond_force_cuda(atom &atoms, float rCut2, float lbox) 
{
	int gridSize;
	int blockSize;
	int minGridSize;
	float milliseconds;

	// timing
	cudaEventRecord(atoms.nonbondStart);

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nonbond_force_kernel, 0, atoms.neighborCount_h[0]); 
    	// Round up according to array size 
    	gridSize = (atoms.neighborCount_h[0] + blockSize - 1) / blockSize; 
	// run nonbond cuda kernel
	nonbond_force_kernel<<<gridSize, blockSize, atoms.nTypes*(atoms.nTypes+1)/2*sizeof(float2)>>>(atoms.pos_d, atoms.for_d, atoms.lj_d, atoms.nAtoms, rCut2, lbox, atoms.neighborList_d, atoms.neighborCount_d, atoms.nTypes);

	// finish timing
	cudaEventRecord(atoms.nonbondStop);
	cudaEventSynchronize(atoms.nonbondStop);
	cudaEventElapsedTime(&milliseconds, atoms.nonbondStart, atoms.nonbondStop);
	return milliseconds;

}

extern "C" void nonbond_force_cuda_grid_block(int nAtoms, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, nonbond_force_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	*gridSize = (nAtoms + *blockSize - 1) / *blockSize; 

}
