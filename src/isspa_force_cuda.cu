#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_vector_routines.h"
#include "isspa_class.h"
#include "isspa_force_cuda.h"
#include "constants.h"

using namespace std;

// constants
__constant__ int nTypes;
__constant__ int nMC;
__constant__ int nGRs;
__constant__ int nAtoms;
__constant__ int nPairs;
__constant__ float2 box;
__constant__ int nRs;
__constant__ float2 forceRparams;
__constant__ float2 gRparams;

// device functions

// CUDA Kernels

__global__ void isspa_force_kernel(float4 *xyz, float *vtot, float *rmax, int *isspaTypes, float *gTable, float *forceTable, float4 *f, curandState *state, float4 *out) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float4 xyz_s[];
	int atom;
	//float rnow;
        //float prob;
       	//float attempt;
	//float x1, x2, r2;
	//float temp;
	//float g1, g2;	//float gnow;
	float dist2, dist;
	float fs;
	float f1, f2, fracDist;
	float vtot_l;
	float rmax_l;
	float2 gRparams_l = gRparams;
	float4 mcpos;
	float4 r;
	float4 mcr;
	int bin;
	int it, jt;
	int i;
	int atom2;
	int igo;
	curandState_t threadState;

	
	// copy atom position to shared memory
	for (i=t;i<nAtoms;i+=blockDim.x) {
		xyz_s[i] = __ldg(xyz+i);
	}
	__syncthreads();
	// move on
	out[index].x = 2.0;

	if (index < nAtoms*nMC)
	{
	        // random number state - store in temporary variable
		threadState = state[index];
		// get atom number of interest
		atom = int(index/(float) nMC);
		mcpos = xyz_s[atom];
		// isspa type
		it = __ldg(isspaTypes+atom);
		vtot_l = __ldg(vtot+it);
		rmax_l = __ldg(rmax+it);
		igo = 0;
		// generate 3D MC pos based inside a sphere rnow
		mcr.x = rmax_l*(2.0f * curand_uniform(&threadState) - 1.0f);
		mcr.y = rmax_l*(2.0f * curand_uniform(&threadState) - 1.0f);
		mcr.z = rmax_l*(2.0f * curand_uniform(&threadState) - 1.0f);
		mcpos += mcr;
		// initialize density to 1.0
		mcpos.w = 1.0;
		// random state in global
		state[index] = threadState;

		
		// Get density from g table  
		// get atom number of interest
		for(atom2=0;atom2<nAtoms;atom2++){
		  if (atom2 != atom){
		    jt = __ldg(isspaTypes + atom2);
		    r = min_image(mcpos - xyz_s[atom2],box.x,box.y);
		    dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
		    dist = sqrtf(dist2);
		    if (dist < rmax_l) {
		      // determine density bin of distance
		      bin = int ( (dist-gRparams_l.x)/gRparams_l.y );
		      // make sure bin is in limits of density table
		      if (bin >= (nGRs-1)) {
			continue;
		      }
		      else if (bin < 0) {
			mcpos.w *= 0.0;
			//break;
		      }
		      else {
			// linearly interpolate between two density bins
			//fracDist = (dist - (gRparams.x+bin*gRparams.y)) / gRparams.y;
			//g1 = __ldg(gTable+jt*nGRs+bin);
			//g2 = __ldg(gTable+jt*nGRs+bin+1);
			//mcpos.w *= g1*(1.0-fracDist)+g2*fracDist;
			mcpos.w *= __ldg(gTable + jt*nGRs+bin);
		      }
		      igo += 1;
		    }		      
		  }
		}
		// add force 
		if (mcpos.w > 0.0f) {
		  // get separation vector
		  // check dist is suppose to be rnow
		  bin = int ( (dist-forceRparams.x)/forceRparams.y + 0.5f);
		  // linearly interpolate between two force bins
		  fracDist = (dist - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
		  f1 = __ldg(forceTable+it*nRs+bin);
		  f2 = __ldg(forceTable+it*nRs+bin+1);
		  fs = f1*(1.0-fracDist)+f2*fracDist;
		  fs *= mcpos.w * vtot_l / igo; 
		  atomicAdd(&(f[atom].x), fs*mcr.x);
		  atomicAdd(&(f[atom].y), fs*mcr.y);
		  atomicAdd(&(f[atom].z), fs*mcr.z);
		}		
	}
}



/* C wrappers for kernels */

float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {

	float milliseconds;

	float4 out_h[nAtoms_h*isspas.nMC];

	
	// timing
	cudaEventRecord(isspas.isspaStart);

	// compute isspa force
	isspa_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize, nAtoms_h*sizeof(float4)>>>(xyz_d, isspas.vtot_d, isspas.rmax_d, isspas.isspaTypes_d, isspas.isspaGTable_d, isspas.isspaForceTable_d, f_d, isspas.randStates_d, isspas.out_d);
	// DEBUG
	cudaMemcpy(out_h, isspas.out_d, nAtoms_h*isspas.nMC*sizeof(float4), cudaMemcpyDeviceToHost);
	//for (int i=0;i<nAtoms_h*isspas.nMC; i++)
	for (int i=0;i<2*isspas.nMC; i++)
	{
	  printf("C %10.6f %10.6f %10.6f %10.6f\n", out_h[i].x, out_h[i].y, out_h[i].z, out_h[i].w);
	}

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

	// fill box with box and half box length
	box_h.x = lbox_h;
	box_h.y = lbox_h/2.0;
	
	// set constant memory
	cudaMemcpyToSymbol(nMC, &isspas.nMC, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &isspas.nTypes, sizeof(int));
	cudaMemcpyToSymbol(nRs, &isspas.nRs, sizeof(int));
	cudaMemcpyToSymbol(nGRs, &isspas.nGRs, sizeof(int));
	cudaMemcpyToSymbol(nAtoms, &nAtoms_h, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &nPairs_h, sizeof(int));
	cudaMemcpyToSymbol(box, &box_h, sizeof(float2));
	cudaMemcpyToSymbol(forceRparams, &isspas.forceRparams, sizeof(float2));
	cudaMemcpyToSymbol(gRparams, &isspas.gRparams, sizeof(float2));

}
