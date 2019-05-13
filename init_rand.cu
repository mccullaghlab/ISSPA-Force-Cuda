/*
 *  routine to initialize random states on device
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "init_rand.cuh"

__global__ void init_rand_states_kernel(curandState *state, int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence 
   	number, no offset */
	curand_init(seed, id, 0, &state[id]);
}

/* C wrappers for kernels */
extern "C" void init_rand_states(curandState *states_d, int seed, int nAtoms)
{
	int minGridSize;
	int blockSize;
	int gridSize;
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_rand_states_kernel, 0, nAtoms); 
    	// Round up according to array size 
    	gridSize = (nAtoms + blockSize - 1) / blockSize; 

	init_rand_states_kernel<<<gridSize, blockSize>>>(states_d, seed);
}



