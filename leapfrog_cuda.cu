
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
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

__global__ void leapfrog_kernel(float *xyz, float *v, float *f, float *mass, float T, float dt, float pnu, int nAtoms, float lbox, long long seed) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float attempt;
	float force;
	float tempMass;
	float tempPos;
	float tempVel;
	int k;
	curandState_t state;

	if (index < nAtoms)
	{
		// initialize random number generator
  		curand_init(seed,index,0,&state);
		attempt = curand_uniform(&state);
		tempMass = __ldg(mass+index);
		// anderson thermostat
		if (attempt < pnu) {
			//thermo_kernel(&v[index*nDim],T,mass[index],index, blockIdx);
			for (k=0;k<nDim;k++) {
				force = __ldg(f+index*nDim+k);
				tempVel = curand_normal(&state) * sqrtf( T / tempMass );
				tempVel += force/tempMass*dt/2.0;
				v[index*nDim+k] = tempVel;
				//xyz[index*nDim+k] += temp*dt;
				tempPos = __ldg(xyz+index*nDim+k);
				tempPos += tempVel*dt;
				if (tempPos > lbox) {
					//xyz[index*nDik] -= (int) (xyz[index*nDim+k]/lbox) * lbox;
					tempPos -= lbox;
				} else if (tempPos < 0.0f) {
					//xyz[index*nDim+k] += (int) (-xyz[index*nDim+k]/lbox+1) * lbox;
					tempPos += lbox;
				}
				xyz[index*nDim+k] = tempPos;
			}
		} else {
			for (k=0;k<nDim;k++) {
				force = __ldg(f+index*nDim+k);
				tempVel = __ldg(v+index*nDim+k);
				tempVel += force/tempMass*dt;
				v[index*nDim+k] = tempVel;
				tempPos = __ldg(xyz+index*nDim+k);
			       	tempPos += tempVel*dt;
				if (tempPos > lbox) {
					tempPos -= lbox;
				} else if (tempPos < 0.0f) {
					tempPos += lbox;
				}
				xyz[index*nDim+k] = tempPos;
			}
		}
	}
}

/* C wrappers for kernels */

extern "C" void leapfrog_cuda(float *xyz_d, float *v_d, float *f_d, float *mass_d, float T, float dt, float pnu, int nAtoms, float lbox, long long seed) {
	int blockSize;      // The launch configurator returned block size
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    	int gridSize;       // The actual grid size needed, based on input size

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, leapfrog_kernel, 0, nAtoms);

    	// Round up according to array size
    	gridSize = (nAtoms + blockSize - 1) / blockSize;
	// run nonbond cuda kernel
	leapfrog_kernel<<<gridSize, blockSize>>>(xyz_d, v_d, f_d, mass_d, T, dt, pnu, nAtoms, lbox, seed);

}
