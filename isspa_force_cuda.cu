
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_vector_routines.h"
#include "isspa_class.h"
#include "isspa_force_cuda.h"
#include "constants.h"

__device__ float atomicMul(float* address, float val) 
{ 
	unsigned int* address_as_u = (unsigned int*)address; 
	unsigned int old = *address_as_u, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_u, assumed, __float_as_uint(val * __uint_as_float(assumed))); 
	} while (assumed != old); return __uint_as_float(old);
}

//__device__ float atomicMul(float* address, float val) 
//{ 
//	int* address_as_int = (int*)address; 
//	int old = *address_as_int, assumed; 
//	do { 
//		assumed = old; 
//		old = atomicCAS(address_as_int, assumed, __float_as_int(val * __float_as_int(assumed))); 
//      } while (assumed != old); return __int_as_float(old);
//}

// CUDA Kernels

__global__ void isspa_mc_kernel(float4 *xyz, float4 *mcpos, float *w, float *x0, int *isspaTypes, int nMC, int nTypes, int nAtoms, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom;
	float rnow;
        float prob;
       	float attempt;
	float x1, x2, r2;
	int it;
	//curandState_t state;

	if (index < nAtoms*nMC)
	{

		// get atom number of interest
		atom = int(index/(float) nMC);
		// isspa type
		it = isspaTypes[atom];
		// select one point from 1D parabolic distribution
		rnow = 1.0f - 2.0f * curand_uniform(&state[index]);
		prob = rnow*rnow;
		attempt = curand_uniform(&state[index]);
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * curand_uniform(&state[index]);
			prob = rnow*rnow;
			attempt = curand_uniform(&state[index]);
		}
		rnow = w[it] * rnow + x0[it];
		// select two points on surface of sphere
		x1 = 1.0f - 2.0f * curand_uniform(&state[index]);
		x2 = 1.0f - 2.0f * curand_uniform(&state[index]);
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f)
		{
			x1 = 1.0f - 2.0f * curand_uniform(&state[index]);
			x2 = 1.0f - 2.0f * curand_uniform(&state[index]);
			r2 = x1*x1 + x2*x2;
		}
		// generate 3D MC pos based on position on surface of sphere and parabolic distribution in depth
		mcpos[index].x = xyz[atom].x + rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mcpos[index].y = xyz[atom].y + rnow*x1*r2;
		mcpos[index].z = xyz[atom].z + rnow*x2*r2;
		// initialize density to 1.0
		mcpos[index].w = 1.0;

	}
}


__global__ void isspa_density_kernel(float4 *xyz, float4 *mcpos, float *x0, float *g0, float *gr2, float *alpha, int *isspaTypes, int nMC, int nTypes, int nAtoms, int nPairs, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float rnow;
	float prob;
	float attempt;
	float4 r;
	int i;
	int atomMC;
	int atom2;
	int mc; 
	int jt;    // atom type of other atom
	float gnow;
	float temp, dist2;
	float hbox;

	if (index < nPairs*nMC)
	{
		hbox = lbox/2.0;
		// get atom number of interest
		atomMC = int(index/(float) nAtoms);
		atom2 = index % (nAtoms);
		jt = __ldg(isspaTypes + atom2);
		r = min_image(mcpos[atomMC] - xyz[atom2],lbox,hbox);
		dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
		if (dist2 < gr2[jt*2]) {
			atomicMul(&(mcpos[atomMC].w),0.0f);
		} else if (dist2 < gr2[jt*2+1]) {
			temp = sqrtf(dist2)-x0[jt];
			gnow = (-alpha[jt] * temp*temp + g0[jt]);
			atomicMul(&(mcpos[atomMC].w),gnow);
		}

	}
}

__global__ void isspa_force_kernel(float4 *xyz, float4 *f, float4 *mcpos, float *vtot, float2 *lj, int *isspaTypes, int nMC, int nTypes, int nAtoms, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom;
	float4 r;
	float4 tempMC;
	float r2, r6;
	float hbox;
	float fs;
	int it;

	if (index < nAtoms*nMC) {
		tempMC = __ldg(mcpos+index);
		if (tempMC.w > 0.0f) {
			hbox = lbox/2.0;
			atom = int(index/(float) nMC);
			it = isspaTypes[atom];
			//r = min_image(tempMC-xyz[atom],lbox,hbox);
			r = tempMC - xyz[atom];
			r2 = r.x*r.x + r.y*r.y + r.z*r.z;
			r6 = powf(r2,-3);
			fs = tempMC.w * r6 * (lj[it].y - lj[it].x * r6)*vtot[it];
			atomicAdd(&(f[atom].x), fs*r.x);
			atomicAdd(&(f[atom].y), fs*r.y);
			atomicAdd(&(f[atom].z), fs*r.z);
		}
	}
}

/* C wrappers for kernels */

float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms, int nPairs, float lbox) {

	float milliseconds;
	float4 mcpos_h[nAtoms*isspas.nMC];
	int i;

	// timing
	cudaEventRecord(isspas.isspaStart);
	

	// generate MC points
	isspa_mc_kernel<<<isspas.mcGridSize, isspas.mcBlockSize>>>(xyz_d, isspas.mcpos_d, isspas.w_d, isspas.x0_d, isspas.isspaTypes_d, isspas.nMC, isspas.nTypes, nAtoms, isspas.randStates_d);
	// compute density at each mc point
	isspa_density_kernel<<<isspas.gGridSize, isspas.gBlockSize>>>(xyz_d, isspas.mcpos_d, isspas.x0_d, isspas.g0_d, isspas.gr2_d, isspas.alpha_d, isspas.isspaTypes_d, isspas.nMC, isspas.nTypes, nAtoms, nPairs, lbox);
	// add to forces
	isspa_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize>>>(xyz_d, f_d, isspas.mcpos_d, isspas.vtot_d, isspas.lj_d, isspas.isspaTypes_d, isspas.nMC, isspas.nTypes, nAtoms, lbox);


	// finish timing
	cudaEventRecord(isspas.isspaStop);
	cudaEventSynchronize(isspas.isspaStop);
	cudaEventElapsedTime(&milliseconds, isspas.isspaStart, isspas.isspaStop);
	return milliseconds;

}

void isspa_grid_block(int nAtoms, int nPairs, isspa& isspas) {

	int minGridSize;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.mcBlockSize, isspa_mc_kernel, 0, nAtoms*isspas.nMC);
    	// Round up according to array size
    	isspas.mcGridSize = (nAtoms*isspas.nMC + isspas.mcBlockSize - 1) / isspas.mcBlockSize;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.gBlockSize, isspa_density_kernel, 0, nPairs*isspas.nMC);
    	// Round up according to array size
    	isspas.gGridSize = (nPairs*isspas.nMC + isspas.gBlockSize - 1) / isspas.gBlockSize;

}
