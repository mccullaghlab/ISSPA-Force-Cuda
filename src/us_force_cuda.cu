
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_vector_routines.h"
#include "us_class.h"
#include "us_force_cuda.h"
#include "constants.h"

// CUDA Kernel

__global__ void us_force_kernel(float4 *xyz, float4 *f, int totalBiasAtoms, int2 *atomList, float *mass, float4 *groupComPos, float *kumb, float x0, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	//unsigned int t = threadIdx.x;
	//extern __shared__ float xyz_s[];
	int2 atom;
	float4 threadXyz;
	float threadMass;
	float4 comSepVec;
	float hbox;
	float prefactor;

	// move on
	if (index < totalBiasAtoms) {
		hbox = lbox/2.0;
		atom = __ldg(atomList + index);
		threadMass = __ldg(mass + index);
		threadXyz = __ldg(xyz + atom.x);
		threadXyz *= threadMass;
		// add to group COM pos - this is likely extremely slow as currently coded
		atomicAdd(&(groupComPos[atom.y].x), threadXyz.x);
		atomicAdd(&(groupComPos[atom.y].y), threadXyz.y);
		atomicAdd(&(groupComPos[atom.y].z), threadXyz.z);
		__syncthreads();
		// compute COM separation vector
		comSepVec = min_image(groupComPos[0] - groupComPos[1], lbox, hbox);
		// magnitude
		comSepVec.w = sqrtf(comSepVec.x*comSepVec.x + comSepVec.y*comSepVec.y + comSepVec.z*comSepVec.z);
		// add to force for atom
		prefactor = kumb[atom.y]*(comSepVec.w-x0)*threadMass/comSepVec.w;
		comSepVec *= prefactor;
		f[atom.x] += comSepVec;
	}

}
/* C wrappers for kernels */

float us_force_cuda(float4 *xyz_d, float4 *f_d, us& bias, float lbox, int nAtoms) {

	float milliseconds;

	// timing
	cudaEventRecord(bias.usStart);

	// zero group COM array on GPU
	cudaMemset(bias.groupComPos_d, 0.0f,  2*sizeof(float4));

	// compute US force and add to atoms
	us_force_kernel<<<bias.usGridSize, bias.usBlockSize>>>(xyz_d, f_d, bias.totalBiasAtoms, bias.atomList_d, bias.mass_d, bias.groupComPos_d, bias.kumb_d, bias.x0, lbox);
	
	// finish timing
	cudaEventRecord(bias.usStop);
	cudaEventSynchronize(bias.usStop);
	cudaEventElapsedTime(&milliseconds, bias.usStart, bias.usStop);
	return milliseconds;

}

void us_grid_block(us& bias) {

	int minGridSize;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bias.usBlockSize, us_force_kernel, 0, bias.totalBiasAtoms);
    	// Round up according to array size
    	bias.usGridSize = (bias.totalBiasAtoms + bias.usBlockSize - 1) / bias.usBlockSize;

}
