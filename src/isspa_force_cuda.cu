
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_vector_routines.h"
#include "isspa_class.h"
#include "isspa_force_cuda.h"
#include "constants.h"

// constants
__constant__ int nTypes;
__constant__ int nMC;
__constant__ int nAtoms;
__constant__ int nPairs;
__constant__ float lbox;
__constant__ int nRs;
__constant__ float2 forceRparams;

// device functions

__device__ float atomicMul(float* address, float val) 
{ 
	unsigned int* address_as_u = (unsigned int*)address; 
	unsigned int old = *address_as_u, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_u, assumed, __float_as_uint(val * __uint_as_float(assumed))); 
	} while (assumed != old); return __uint_as_float(old);
}


// CUDA Kernels

__global__ void isspa_mc_kernel(float4 *xyz, float4 *mcpos, float2 *x0_w, int *isspaTypes, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float2 params_s[];
	int atom;
	float rnow;
        float prob;
       	float attempt;
	float x1, x2, r2;
	float4 pos;
	int it;
	int i;
	curandState_t threadState;

	// copy density parameters to shared memory
	for (i=t;i<nTypes;i+=blockDim.x) {
		params_s[i] = __ldg(x0_w+i);
	}
	__syncthreads();
	// move on
	if (index < nAtoms*nMC)
	{

		// random number state - store in temporary variable
		threadState = state[index];
		// get atom number of interest
		atom = int(index/(float) nMC);
		pos = __ldg(xyz+atom);
		// isspa type
		it = __ldg(isspaTypes+atom);
		// select one point from 1D parabolic distribution
		rnow = 1.0f - 2.0f * curand_uniform(&threadState);
		prob = rnow*rnow;
		attempt = curand_uniform(&threadState);
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * curand_uniform(&threadState);
			prob = rnow*rnow;
			attempt = curand_uniform(&threadState);
		}
		rnow = params_s[it].y * rnow + params_s[it].x;
		// select two points on surface of sphere
		x1 = 1.0f - 2.0f * curand_uniform(&threadState);
		x2 = 1.0f - 2.0f * curand_uniform(&threadState);
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f)
		{
			x1 = 1.0f - 2.0f * curand_uniform(&threadState);
			x2 = 1.0f - 2.0f * curand_uniform(&threadState);
			r2 = x1*x1 + x2*x2;
		}
		// generate 3D MC pos based on position on surface of sphere and parabolic distribution in depth
		mcpos[index].x = pos.x + rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mcpos[index].y = pos.y + rnow*x1*r2;
		mcpos[index].z = pos.z + rnow*x2*r2;
		// initialize density to 1.0
		mcpos[index].w = 1.0;

		// random state in global
		state[index] = threadState;

	}
}


__global__ void isspa_density_kernel(float4 *xyz, float4 *mcpos, float *x0, float4 *gr2_g0_alpha, int *isspaTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float4 gr2_g0_alpha_s[];
	float4 r;
	int i;
	int atomMC;
	int atom2;
	int jt;    // atom type of other atom
	float gnow;
	float temp, dist2;
	float hbox;

	// copy density parameters to shared memory
	for (i=t;i<nTypes;i+=blockDim.x) {
		gr2_g0_alpha_s[i] = __ldg(gr2_g0_alpha+i);
	}
	__syncthreads();
	// move on
	if (index < nPairs*nMC)
	{
		hbox = lbox/2.0;
		// get atom number of interest
		atomMC = int(index/(float) nAtoms);
		atom2 = index % (nAtoms);
		jt = __ldg(isspaTypes + atom2);
		r = min_image(__ldg(mcpos+atomMC) - __ldg(xyz+atom2),lbox,hbox);
		dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
		if (dist2 < gr2_g0_alpha_s[jt].x) {
			atomicMul(&(mcpos[atomMC].w),0.0f);
		} else if (dist2 < gr2_g0_alpha_s[jt].y) {
			temp = sqrtf(dist2)-__ldg(x0+jt);
			gnow = (-gr2_g0_alpha_s[jt].w * temp*temp + gr2_g0_alpha_s[jt].z);
			atomicMul(&(mcpos[atomMC].w),gnow);
		}

	}
}

__global__ void isspa_force_kernel(float4 *xyz, float4 *f, float4 *mcpos, float *vtot, int *isspaTypes, float *forceTable) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float vtot_s[];
	int atom;
	float4 r;
	float4 tempMC;
	float r2, dist;
	float fs;
	float f1, f2, fracDist;
	int it;
	int bin;
	int i;

	// copy density parameters to shared memory
	for (i=t;i<nTypes;i+=blockDim.x) {
		vtot_s[i] = __ldg(vtot+i);
	}
	__syncthreads();
	// move on
	if (index < nAtoms*nMC) {
		tempMC = __ldg(mcpos+index);
		if (tempMC.w > 0.0f) {
			atom = int(index/(float) nMC);
			it = __ldg(isspaTypes+atom);
			// get separation vector
			r = tempMC - __ldg(xyz+atom);
			r2 = r.x*r.x + r.y*r.y + r.z*r.z;
			dist = sqrtf(r2);
			bin = int ( (dist-forceRparams.x)/forceRparams.y + 0.5f);
			// linearly interpolate between two force bins
			fracDist = (dist - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
			f1 = __ldg(forceTable+it*nRs+bin);
			f2 = __ldg(forceTable+it*nRs+bin+1);
			fs = f1*(1.0-fracDist)+f2*fracDist;
			fs *= tempMC.w * vtot_s[it];
			atomicAdd(&(f[atom].x), fs*r.x);
			atomicAdd(&(f[atom].y), fs*r.y);
			atomicAdd(&(f[atom].z), fs*r.z);
		}
	}
}
/*
__global__ void isspa_mc_density_force_kernel(float4 *xyz, float4 *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *vtot, float2 *lj, int *isspaTypes, int nMC, int nTypes, int nAtoms, curandState *state, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom, atom2;
	float rnow;
        float prob;
       	float attempt;
	float4 mcpos;
	float4 r;
	float x1, x2, r2;
	int it, jt;
	float temp;
	float hbox;
	float dist2;
	float rinv;
	float r6;
	float fs;
	int ev_flag;
	curandState_t threadState;

	if (index < nAtoms*nMC)
	{
		// random number state - store in temporary variable
		threadState = state[index];
		// get atom number of interest
		atom = int(index/(float) nMC);
		// isspa type
		it = isspaTypes[atom];
		// select one point from 1D parabolic distribution
		rnow = 1.0f - 2.0f * curand_uniform(&threadState);
		prob = rnow*rnow;
		attempt = curand_uniform(&threadState);
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * curand_uniform(&threadState);
			prob = rnow*rnow;
			attempt = curand_uniform(&threadState);
		}
		rnow = w[it] * rnow + x0[it];
		// select two points on surface of sphere
		x1 = 1.0f - 2.0f * curand_uniform(&threadState);
		x2 = 1.0f - 2.0f * curand_uniform(&threadState);
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f)
		{
			x1 = 1.0f - 2.0f * curand_uniform(&threadState);
			x2 = 1.0f - 2.0f * curand_uniform(&threadState);
			r2 = x1*x1 + x2*x2;
		}
		// generate 3D MC pos based on position on surface of sphere and parabolic distribution in depth
		mcpos.x = xyz[atom].x + rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mcpos.y = xyz[atom].y + rnow*x1*r2;
		mcpos.z = xyz[atom].z + rnow*x2*r2;
		// initialize density to 1.0
		mcpos.w = 1.0;

		ev_flag = 0;
		for (atom2=0;atom2<nAtoms;atom2++)
		{
			jt = isspaTypes[atom2];
			r = min_image(mcpos - __ldg(xyz+atom2),lbox,hbox);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			if (dist2 < gr2[jt*2]) {
				ev_flag = 1;
				break;
			} else if (dist2 < gr2[jt*2+1]) {
				temp = sqrtf(dist2)-x0[jt];
				mcpos.w *= (-alpha[jt] * temp*temp + g0[jt]);
			}
		}

		if (ev_flag ==0) {
			rinv = 1.0f / rnow;
			r2 = rinv * rinv;
			r6 = r2 * r2 * r2;
			fs = mcpos.w * r6 * (lj[it].y - lj[it].x * r6)*vtot[it];
			atomicAdd(&(f[atom].x), fs*r.x);
			atomicAdd(&(f[atom].y), fs*r.y);
			atomicAdd(&(f[atom].z), fs*r.z);
		}
		// random state in global
		state[index] = threadState;

	}
}
*/
/* C wrappers for kernels */

float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas) {

	float milliseconds;

	// timing
	cudaEventRecord(isspas.isspaStart);
	

	// generate MC points
	isspa_mc_kernel<<<isspas.mcGridSize, isspas.mcBlockSize,isspas.nTypes*sizeof(float2)>>>(xyz_d, isspas.mcpos_d, isspas.x0_w_d, isspas.isspaTypes_d, isspas.randStates_d);
	//printf("MC points generated\n");
	// compute density at each mc point
	isspa_density_kernel<<<isspas.gGridSize, isspas.gBlockSize,isspas.nTypes*sizeof(float4)>>>(xyz_d, isspas.mcpos_d, isspas.x0_d, isspas.gr2_g0_alpha_d, isspas.isspaTypes_d);
	// add to forces
	isspa_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize,isspas.nTypes*sizeof(float)>>>(xyz_d, f_d, isspas.mcpos_d, isspas.vtot_d, isspas.isspaTypes_d, isspas.isspaForceTable_d);


	// finish timing
	cudaEventRecord(isspas.isspaStop);
	cudaEventSynchronize(isspas.isspaStop);
	cudaEventElapsedTime(&milliseconds, isspas.isspaStart, isspas.isspaStop);
	return milliseconds;

}

void isspa_grid_block(int nAtoms_h, int nPairs_h, float lbox_h, isspa& isspas) {

	int minGridSize;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.mcBlockSize, isspa_mc_kernel, 0, nAtoms_h*isspas.nMC);
    	// Round up according to array size
    	isspas.mcGridSize = (nAtoms_h*isspas.nMC + isspas.mcBlockSize - 1) / isspas.mcBlockSize;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.gBlockSize, isspa_density_kernel, 0, nPairs_h*isspas.nMC);
    	// Round up according to array size
    	isspas.gGridSize = (nPairs_h*isspas.nMC + isspas.gBlockSize - 1) / isspas.gBlockSize;

	// set constant memory
	cudaMemcpyToSymbol(nMC, &isspas.nMC, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &isspas.nTypes, sizeof(int));
	cudaMemcpyToSymbol(nRs, &isspas.nRs, sizeof(int));
	cudaMemcpyToSymbol(nAtoms, &nAtoms_h, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &nPairs_h, sizeof(int));
	cudaMemcpyToSymbol(lbox, &lbox_h, sizeof(float));
	cudaMemcpyToSymbol(forceRparams, &isspas.forceRparams, sizeof(float2));


}
