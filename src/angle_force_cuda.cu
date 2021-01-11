
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_vector_routines.h"
#include "angle_class.h"
#include "angle_force_cuda.h"
#include "constants.h"

// CUDA Kernels

__global__ void angle_force_kernel(float4 *xyz, float4 *f, int nAtoms, float lbox, int4 *angleAtoms, float2 *angleParams, int nAngles) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int4 atoms;
	float4 r1;
	float4 r2;
	float c11, c22, c12;
	float b;
	float theta;
	float fang;
	float hbox;
	float2 params;
	

	if (index < nAngles)
	{
		hbox = 0.5f*lbox;
		// determine atoms to work on
		atoms = __ldg(angleAtoms+index);
		// get distance vectors separating the atoms
		r1 = min_image(__ldg(xyz+atoms.x) - __ldg(xyz+atoms.y),lbox,hbox);
		r2 = min_image(__ldg(xyz+atoms.y) - __ldg(xyz+atoms.z),lbox,hbox);
		// compute dot products
		c11 = r1.x*r1.x + r1.y*r1.y + r1.z*r1.z;
		c22 = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
		c12 = r1.x*r2.x + r1.y*r2.y + r1.z*r2.z;
		b = -__fdividef(c12,sqrtf(c11*c22));
		// make sure b is in the domain of the arccos
		if (b>=1.0f) {
			// theta is zero
			theta = 1.0e-16f;
		} else if (b <= -1.0f) {
			// theta is pi
			theta = PI;
		} else {
			// b is in domain so take arccos
			theta = acosf(b);
		}
		// grab parameters for angle atoms type - stored as fourth integer in angleAtoms
		params = __ldg(angleParams+atoms.w);
		// compute force component
		fang = __fdividef(params.x*(theta - params.y),sqrtf(c11*c22-c12*c12));
		// atomicAdd forces to each atom
		atomicAdd(&(f[atoms.x].x), fang*(c12/c11*r1.x-r2.x));
		atomicAdd(&(f[atoms.y].x), fang*((1.0f+c12/c22)*r2.x-(1.0f+c12/c11)*r1.x));
		atomicAdd(&(f[atoms.z].x), fang*(r1.x-c12/c22*r2.x));
		atomicAdd(&(f[atoms.x].y), fang*(c12/c11*r1.y-r2.y));
		atomicAdd(&(f[atoms.y].y), fang*((1.0f+c12/c22)*r2.y-(1.0f+c12/c11)*r1.y));
		atomicAdd(&(f[atoms.z].y), fang*(r1.y-c12/c22*r2.y));
		atomicAdd(&(f[atoms.x].z), fang*(c12/c11*r1.z-r2.z));
		atomicAdd(&(f[atoms.y].z), fang*((1.0f+c12/c22)*r2.z-(1.0f+c12/c11)*r1.z));
		atomicAdd(&(f[atoms.z].z), fang*(r1.z-c12/c22*r2.z));

	}
}

/* C wrappers for kernels */

float angle_force_cuda(float4 *xyz_d, float4 *f_d, int nAtoms, float lbox, angle& angles) 
{
	float milliseconds;
	// initialize timing stuff
	cudaEventRecord(angles.angleStart);
	
	// run angle cuda kernel
	angle_force_kernel<<<angles.gridSize, angles.blockSize>>>(xyz_d, f_d, nAtoms, lbox, angles.angleAtoms_d, angles.angleParams_d, angles.nAngles);

	// finalize timing
	cudaEventRecord(angles.angleStop);
	cudaEventSynchronize(angles.angleStop);
	cudaEventElapsedTime(&milliseconds, angles.angleStart, angles.angleStop);

	// return time
	return milliseconds;

}

void angle_force_cuda_grid_block(int nAngles, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, angle_force_kernel, 0, nAngles); 

    	// Round up according to array size 
    	*gridSize = (nAngles + *blockSize - 1) / *blockSize; 
}
