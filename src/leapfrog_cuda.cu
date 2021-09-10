
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "config_class.h"
#include "leapfrog_cuda.h"
//#include "constants_cuda.cuh"
#include "constants.h"

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// CUDA Kernels

/*__device__ void thermo_kernel( float v*, float T, float mass, int atom, int block) {

	float rang;
	curandState_t state;
        curand_init(0,blockIdx.x,atom,&state);
	rang = curand_normal(&state);
	v[0] = rang*sqrtf(T/mass);
	rang = curand_normal(&state);
	v[1] = rang*sqrtf(T/mass);
	rang = curand_normal(&state);
	v[2] = rang*sqrtf(T/mass);

}*/

__global__ void leapfrog_kernel(float4 *xyz, float4 *v, float4 *f, float T, float dt, float pnu, int nAtoms, float lbox, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float attempt;
	float4 force;
	float4 tempPos;
	float4 tempVel;
	//curandState_t state;

	if (index < nAtoms)
	{
		// generate random number
		attempt = curand_uniform(&state[index]);
		// load values
		force = __ldg(f+index);
		tempVel = __ldg(v+index);
		tempPos = __ldg(xyz+index);
                //printf("index: %d mass: %f\n",index,tempVel.w);

		// anderson thermostat
		if (attempt < pnu) {
			tempVel.x = curand_normal(&state[index]) * sqrtf( T / tempVel.w );
			tempVel.y = curand_normal(&state[index]) * sqrtf( T / tempVel.w );
			tempVel.z = curand_normal(&state[index]) * sqrtf( T / tempVel.w );
			tempVel += force *__fdividef(dt,tempVel.w*2.0f);
			tempPos += tempVel*dt;
		} else {
			tempVel += force * __fdividef(dt,tempVel.w);
			tempPos += tempVel*dt;
		}
		// save new velocities and positions to global memory
		v[index] = tempVel;
		//xyz[index] = wrap(tempPos,lbox);
                xyz[index] = tempPos;

	}
}

/* C wrappers for kernels */

//extern "C" void leapfrog_cuda(float *xyz_d, float *v_d, float *f_d, float *mass_d, float T, float dt, float pnu, int nAtoms, float lbox, curandState *randStates_d) {
float leapfrog_cuda(atom& atoms, config& configs)
{
	float milliseconds;

	// timing
	cudaEventRecord(atoms.leapFrogStart);
	// run nonbond cuda kernel
	leapfrog_kernel<<<atoms.gridSize, atoms.blockSize>>>(atoms.pos_d, atoms.vel_d, atoms.for_d, configs.T, configs.dt, configs.pnu, atoms.nAtoms, configs.lbox, atoms.randStates_d);
	// finalize timing
	cudaEventRecord(atoms.leapFrogStop);
	cudaEventSynchronize(atoms.leapFrogStop);
	cudaEventElapsedTime(&milliseconds, atoms.leapFrogStart, atoms.leapFrogStop);
	return milliseconds;
}

extern "C" void leapfrog_cuda_grid_block(int nAtoms, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, leapfrog_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	*gridSize = (nAtoms + *blockSize - 1) / *blockSize; 

}

