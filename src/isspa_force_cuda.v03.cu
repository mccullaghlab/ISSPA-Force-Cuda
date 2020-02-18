
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
__constant__ float2 box;
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

__global__ void isspa_force_kernel(float4 *xyz, float4 *f, float2 *x0_w, float4 *gr2_g0_alpha, float *vtot, int *isspaTypes, float *forceTable, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float2 params_s[];
	int atom;
	float rnow;
        float prob;
       	float attempt;
	float x1, x2, r2;
	float4 mcpos;
	float4 r;
	float dist2, dist;
	int bin;
	int it, jt;
	int i;
	int atom2;
	float fs;
	float f1, f2, fracDist;
	float temp, gnow;
	float4 gr2_g0_alpha_l;
	curandState_t threadState;

	// copy density parameters to shared memory
	for (i=t;i<nTypes;i+=blockDim.x) {
		params_s[i] = __ldg(x0_w+i);
	}
	__syncthreads();
	// move on
	if (index < nAtoms*nMC)
	{
		// local variables
		// random number state - store in temporary variable
		threadState = state[index];
		// get atom number of interest
		atom = int(index/(float) nMC);
		mcpos = __ldg(xyz+atom);
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
		mcpos.x = mcpos.x + rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mcpos.y = mcpos.y + rnow*x1*r2;
		mcpos.z = mcpos.z + rnow*x2*r2;
		// initialize density to 1.0
		mcpos.w = 1.0;
		// random state in global
		state[index] = threadState;

		// determine density at MC pos
		for (atom2=0;atom2<nAtoms;atom2++) {
			if (atom2 != atom) {	
				jt = __ldg(isspaTypes + atom2);
				gr2_g0_alpha_l = __ldg(gr2_g0_alpha+jt);
				r = min_image(mcpos - __ldg(xyz+atom2),box.x,box.y);
				dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
				if (dist2 < gr2_g0_alpha_l.x) {
					mcpos.w = 0.0f;
				} else if (dist2 < gr2_g0_alpha_l.y) {
					temp = sqrtf(dist2)-params_s[jt].x;
					gnow = (-gr2_g0_alpha_l.w * temp*temp + gr2_g0_alpha_l.z);
					mcpos.w *= gnow;
				}
			}
		}

		// add force to atoms
		if (mcpos.w > 0.0f) {
			// get separation vector
			r = mcpos - __ldg(xyz+atom);
			r2 = r.x*r.x + r.y*r.y + r.z*r.z;
			dist = sqrtf(r2);
			bin = int ( (dist-forceRparams.x)/forceRparams.y + 0.5f);
			// linearly interpolate between two force bins
			fracDist = (dist - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
			f1 = __ldg(forceTable+it*nRs+bin);
			f2 = __ldg(forceTable+it*nRs+bin+1);
			fs = f1*(1.0-fracDist)+f2*fracDist;
			fs *= mcpos.w * vtot[it];
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
	isspa_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize,isspas.nTypes*sizeof(float2)>>>(xyz_d, f_d, isspas.x0_w_d, isspas.gr2_g0_alpha_d, isspas.vtot_d, isspas.isspaTypes_d, isspas.isspaForceTable_d, isspas.randStates_d);


	// finish timing
	cudaEventRecord(isspas.isspaStop);
	cudaEventSynchronize(isspas.isspaStop);
	cudaEventElapsedTime(&milliseconds, isspas.isspaStart, isspas.isspaStop);
	return milliseconds;

}

void isspa_grid_block(int nAtoms_h, int nPairs_h, float lbox_h, isspa& isspas) {

	int minGridSize;
	float2 box_h;

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.mcBlockSize, isspa_force_kernel, 0, nAtoms_h*isspas.nMC);
    	// Round up according to array size
    	isspas.mcGridSize = (nAtoms_h*isspas.nMC + isspas.mcBlockSize - 1) / isspas.mcBlockSize;


	// set constant memory
	cudaMemcpyToSymbol(nMC, &isspas.nMC, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &isspas.nTypes, sizeof(int));
	cudaMemcpyToSymbol(nRs, &isspas.nRs, sizeof(int));
	cudaMemcpyToSymbol(nAtoms, &nAtoms_h, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &nPairs_h, sizeof(int));
	box_h.x = lbox_h;
	box_h.y = lbox_h/2.0;
	cudaMemcpyToSymbol(box, &box_h, sizeof(float2));
	cudaMemcpyToSymbol(forceRparams, &isspas.forceRparams, sizeof(float2));


}
