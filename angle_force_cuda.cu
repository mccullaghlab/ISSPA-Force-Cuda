
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "angle_class.h"
#include "angle_force_cuda.h"
#include "constants.h"

// CUDA Kernels

__global__ void angle_force_kernel(float *xyz, float *f, int nAtoms, float lbox, int *angleAtoms, float *angleKs, float *angleX0s, int nAngles) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
//	extern __shared__ float xyz_s[];
//	extern __shared__ int angleAtoms_s[];
	int atom1;
	int atom2;
	int atom3;
	int k;
	float r1[nDim];
	float r2[nDim];
	float c11, c22, c12;
	float b;
	float theta;
	float fang;
	float hbox;
	

	if (index < nAngles)
	{
		hbox = lbox/2.0;
		// determine two atoms to work  - these will be unique to each index
		atom1 = __ldg(angleAtoms+index*3);
		atom2 = __ldg(angleAtoms+index*3+1);
		atom3 = __ldg(angleAtoms+index*3+2);
		c11 = 0.0f;
		c22 = 0.0f;
		c12 = 0.0f;
		for (k=0;k<nDim;k++) {
			r1[k] = __ldg(xyz+atom1+k) - __ldg(xyz+atom2+k);
			r2[k] = __ldg(xyz+atom2+k) - __ldg(xyz+atom3+k);
			// assuming no more than one box away
			if (r1[k] > hbox) {
				r1[k] -= lbox;
			} else if (r1[k] < -hbox) {
				r1[k] += lbox;
			}
			if (r2[k] > hbox) {
				r2[k] -= lbox;
			} else if (r2[k] < -hbox) {
				r2[k] += lbox;
			}
			c11 += r1[k]*r1[k];
			c22 += r2[k]*r2[k];
			c12 += r1[k]*r2[k];
		}
		b = -c12/sqrtf(c11*c22);
		// make sure b is in the domain of the arccos
		if (b>=1.0f) {
			// theta is zero
			theta = 1.0e-16;
		} else if (b <= -1.0f) {
			// theta is pi
			theta = PI;
		} else {
			// b is in domain so take arccos
			theta = acos(b);
		}
		fang = angleKs[index]*(theta - angleX0s[index])/sqrtf(c11*c22-c12*c12);
		for (k=0;k<3;k++) {
			atomicAdd(&f[atom1+k], fang*(c12/c11*r1[k]-r2[k]));
			atomicAdd(&f[atom2+k], fang*((1.0f+c12/c22)*r2[k]-(1.0f+c12/c11)*r1[k]));
			atomicAdd(&f[atom3+k], fang*(r1[k]-c12/c22*r2[k]));
		}

	}
}

/* C wrappers for kernels */

float angle_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, angle& angles) 
{
	float milliseconds;
	// initialize timing stuff
	cudaEventRecord(angles.angleStart);

	// run nonangle cuda kernel
	angle_force_kernel<<<angles.gridSize, angles.blockSize>>>(xyz_d, f_d, nAtoms, lbox, angles.angleAtoms_d, angles.angleKs_d, angles.angleX0s_d, angles.nAngles);
	// finalize timing
	cudaEventRecord(angles.angleStop);
	cudaEventSynchronize(angles.angleStop);
	cudaEventElapsedTime(&milliseconds, angles.angleStart, angles.angleStop);
	return milliseconds;

}

void angle_force_cuda_grid_block(int nAngles, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, angle_force_kernel, 0, nAngles); 

    	// Round up according to array size 
    	*gridSize = (nAngles + *blockSize - 1) / *blockSize; 
}
