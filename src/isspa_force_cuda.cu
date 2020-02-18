
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


// CUDA Kernels

__global__ void isspa_force_kernel(float4 *xyz, float4 *f, float4 *x0_w_vtot, float4 *gr2_g0_alpha, int *isspaTypes, float *forceTable, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float4 xyz_s[];
	int atom;
	float rnow;
        float prob;
       	float attempt;
	float x1, x2, r2;
	float4 mcpos;
	float4 mcr;
	float4 r;
	float dist2;
	int bin;
	int it, jt;
	int i;
	int atom2;
	float fs;
	float f1, f2, fracDist;
	float temp, gnow;
	float4 gr2_g0_alpha_l;
	float4 x0_w_vtot_l;
	curandState_t threadState;

	// copy atom positions to shared memory
	for (i=t;i<nAtoms;i+=blockDim.x) {
		xyz_s[i] = __ldg(xyz+i);
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
		mcpos = xyz_s[atom];
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
		x0_w_vtot_l = __ldg(x0_w_vtot+it);
		rnow = x0_w_vtot_l.y * rnow + x0_w_vtot_l.x;
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
		mcr.x = rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mcr.y = rnow*x1*r2;
		mcr.z = rnow*x2*r2;
		mcpos += mcr;
		// initialize density to 1.0
		mcpos.w =  x0_w_vtot_l.z;
		// random state in global
		state[index] = threadState;

		// determine density at MC pos
		for (atom2=0;atom2<nAtoms;atom2++) {
			if (atom2 != atom) {	
				jt = __ldg(isspaTypes + atom2);
				gr2_g0_alpha_l = __ldg(gr2_g0_alpha+jt);
				r = min_image(mcpos - xyz_s[atom2],box.x,box.y);
				dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
				if (dist2 < gr2_g0_alpha_l.x) {
					mcpos.w = 0.0f;
					break;
				} else if (dist2 < gr2_g0_alpha_l.y) {
					temp = sqrtf(dist2)-__ldg(x0_w_vtot+jt).x;
					gnow = (-gr2_g0_alpha_l.w * temp*temp + gr2_g0_alpha_l.z);
					mcpos.w *= gnow;
				}
			}
		}

		// add force to atoms
		if (mcpos.w > 0.0f) {
			bin = int ( (rnow-forceRparams.x)/forceRparams.y + 0.5f);
			// linearly interpolate between two force bins
			fracDist = (rnow - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
			f1 = __ldg(forceTable+it*nRs+bin);
			f2 = __ldg(forceTable+it*nRs+bin+1);
			fs = f1*(1.0-fracDist)+f2*fracDist;
			fs *= mcpos.w;
			atomicAdd(&(f[atom].x), fs*mcr.x);
			atomicAdd(&(f[atom].y), fs*mcr.y);
			atomicAdd(&(f[atom].z), fs*mcr.z);
		}

	}
}


// CUDA Kernel wrapper
float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {

	float milliseconds;

	// timing
	cudaEventRecord(isspas.isspaStart);
	

	// generate MC points
	isspa_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize,nAtoms_h*sizeof(float4)>>>(xyz_d, f_d, isspas.x0_w_vtot_d, isspas.gr2_g0_alpha_d, isspas.isspaTypes_d, isspas.isspaForceTable_d, isspas.randStates_d);


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
