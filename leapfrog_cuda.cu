
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
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

__global__ void leapfrog_kernel(float *xyz, float *v, float *f, float *mass, float T, float dt, float pnu, int nAtoms, float lbox, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float attempt;
	float force;
	float tempMass;
	float tempPos;
	float tempVel;
	int k;
	//curandState_t state;

	if (index < nAtoms)
	{
		// initialize random number generator
  		//curand_init(seed,index,0,&state);
		attempt = curand_uniform(&state[index]);
		tempMass = __ldg(mass+index);
		// anderson thermostat
		if (attempt < pnu) {
			//thermo_kernel(&v[index*nDim],T,mass[index],index, blockIdx);
			for (k=0;k<nDim;k++) {
				force = __ldg(f+index*nDim+k);
				tempVel = curand_normal(&state[index]) * sqrtf( T / tempMass );
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

//extern "C" void leapfrog_cuda(float *xyz_d, float *v_d, float *f_d, float *mass_d, float T, float dt, float pnu, int nAtoms, float lbox, curandState *randStates_d) {
float leapfrog_cuda(atom& atoms, config& configs)
{
	float milliseconds;

	// timing
	cudaEventRecord(atoms.leapFrogStart);
	// run nonbond cuda kernel
	leapfrog_kernel<<<atoms.gridSize, atoms.blockSize>>>(atoms.xyz_d, atoms.v_d, atoms.f_d, atoms.mass_d, configs.T, configs.dt, configs.pnu, atoms.nAtoms, configs.lbox, atoms.randStates_d);
	// finalize timing
	cudaEventRecord(atoms.leapFrogStop);
	cudaEventSynchronize(atoms.leapFrogStop);
	cudaEventElapsedTime(&milliseconds, atoms.leapFrogStart, atoms.leapFrogStop);
	return milliseconds;
}
