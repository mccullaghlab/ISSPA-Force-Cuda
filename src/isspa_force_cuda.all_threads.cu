
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
__constant__ float mu;
__constant__ float rho;
__constant__ float lbox;
__constant__ float hbox;
__constant__ int nForceRs;
__constant__ float2 forceRparams;
__constant__ int nGRs;
__constant__ float2 gRparams;
__constant__ float maxForce;

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

__global__ void isspa_mc_kernel(float4 *xyz, float4 *mcPos, float4 *mcDist, int *isspaTypes, float *forceTable, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float4 mcDist_s[];
	int atom;
	float rnow;
	float x1, x2, r2;
	float4 atomPos;
	float4 tempMCPos;
	int it;
	int i;
	int bin;
	float fs;
	float2 forceRparams_l = forceRparams;
	int localnForceRs = nForceRs;
	int localnAtoms = nAtoms;
	int localnMC = nMC;
	curandState_t threadState;

	// copy MC distribution parameters to shared memory
	for (i=t;i<nTypes;i+=blockDim.x) {
		mcDist_s[i] = __ldg(mcDist+i);
	}
	__syncthreads();
	// move on
	if (index < localnAtoms*localnMC)
	{

		// random number state - store in temporary variable
		threadState = state[index];
		// get atom number of interest
		atom = int(index/(float) localnMC);
		atomPos = __ldg(xyz+atom);
		// isspa type
		it = __ldg(isspaTypes+atom);
		// select uniform number between rmin and rmax of MC dist
		rnow = mcDist_s[it].x + mcDist_s[it].z * curand_uniform(&threadState);
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
		// generate 3D MC pos based on position 
		tempMCPos.x =  rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		tempMCPos.y = rnow*x1*r2;
		tempMCPos.z = rnow*x2*r2;
		// save force between MC point and atom of interest
		bin = int ( (rnow-forceRparams_l.x)/forceRparams_l.y );
		if (bin >= (localnForceRs-1)) {
			fs = 0.0;
		} else {
			// linearly interpolate between two force bins
			//fracDist = (dist - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
			//f1 = __ldg(forceTable+it*nForceRs+bin);
			//f2 = __ldg(forceTable+it*nForceRs+bin+1);
			//fs = f1*(1.0-fracDist)+f2*fracDist;
			fs = __ldg(forceTable+it*localnForceRs+bin);
		}
		// add atom position to mc point position
		mcPos[index] = tempMCPos + atomPos;
		// initialize density to N/P(r) = rho*4*pi*(r2-r1)*r^2/nMC
		// note should be dividing fs by rnow and then multiplying my rnow^2 so just multiply by rnow
		mcPos[index].w = fs*mcDist_s[it].w*rnow;
		// random state in global
		state[index] = threadState;

	}
}


__global__ void isspa_density_kernel(float4 *xyz, float4 *mcPos, int *isspaTypes,  float *gTable) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float4 r;
	float4 mcPos_l;
	int atomMC;
	int atom2;
	int jt;    // atom type of other atom
	int bin;
	float fracDist;
	float gnow;
	float dist, dist2;
	float g1, g2;
	float hbox_l = hbox;
	float lbox_l = lbox;
	float2 gRparams_l = gRparams;
	int localnGRs=nGRs;
	int localnAtoms = nAtoms;
	int localnMC = nMC;
	int localnPairs = nPairs;

	// thread for each MC-atom pair
	if (index < localnPairs*localnMC)
	{
		// get atom number of interest
		atom2  = int(index/(float) (localnAtoms*localnMC));
		// get MC point index of interest
		atomMC = index % (localnAtoms*localnMC);
		// get ISSPA atom type of atom of interest
		jt = __ldg(isspaTypes + atom2);
		// determine separation vector between MC point and atom
		mcPos_l = __ldg(mcPos+atomMC);
		r = min_image(mcPos_l - __ldg(xyz+atom2),lbox_l,hbox_l);
		dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
		dist = sqrtf(dist2);
		// deterrmine density bin of distance
		bin = int ( (dist-gRparams_l.x)/gRparams_l.y );
		// make sure bin is in limits of density table
		if (bin >= (localnGRs-1)) {
			gnow = 1.0;
		} else if (bin < 0) {
			gnow = 0.0;
			atomicMul(&(mcPos[atomMC].w),gnow);
		//} else if (fabsf(mcPos_l.w) > 1.0E-30)  {
		} else {
			// linearly interpolate between two density bins
			//fracDist = (dist - (gRparams.x+bin*gRparams.y)) / gRparams.y;
			//g1 = __ldg(gTable+jt*nGRs+bin);
			//g2 = __ldg(gTable+jt*nGRs+bin+1);
			//gnow = g1*(1.0-fracDist)+g2*fracDist;
			gnow = __ldg(gTable + jt*localnGRs+bin);
			atomicMul(&(mcPos[atomMC].w),gnow);
		}

	}
}

__global__ void isspa_force_kernel(float4 *xyz, float4 *f, float4 *mcPos) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom;
	float4 tempMC;
	float4 tempAtom;
	float4 r;
	float maxForce_l = maxForce;
	int nAtoms_l = nAtoms;
	int nMC_l = nMC;

	// thread for each MC point
	if (index < nAtoms_l*nMC_l) {
		// grab MC position 
		tempMC = __ldg(mcPos+index);
		//if (fabsf(tempMC.w) > 1.0E-30) {
			// figure out which atom we are working on
			atom = int(index/(float) nMC_l);
			// get separation vector
			tempAtom = __ldg(xyz + atom);
			r = tempAtom - tempMC;
			// make sure force is within reasonable threshold
	 		if ( tempMC.w > maxForce_l) {		
				tempMC.w = maxForce_l;
			}
			// multiply force by density (and normalization factor)
			r *= tempMC.w;
			// add force to atom
			atomicAdd(&(f[atom].x), r.x);
			atomicAdd(&(f[atom].y), r.y);
			atomicAdd(&(f[atom].z), r.z);
		//}
	}
}

/* C wrappers for kernels */

float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {

	float milliseconds;
	//float4 mcPos_h[nAtoms_h*isspas.nMC];
	//FILE *posFile = fopen("mcPosTemp.xyz", "w");
	//int i;

	// timing
	cudaEventRecord(isspas.isspaStart);
	

	// generate MC points
	isspa_mc_kernel<<<isspas.mcGridSize, isspas.mcBlockSize,isspas.nTypes*sizeof(float4)>>>(xyz_d, isspas.mcPos_d, isspas.mcDist_d, isspas.isspaTypes_d, isspas.isspaForceTable_d, isspas.randStates_d);
	// compute density at each mc point
	isspa_density_kernel<<<isspas.gGridSize, isspas.gBlockSize>>>(xyz_d, isspas.mcPos_d, isspas.isspaTypes_d, isspas.isspaGTable_d);
	// DEBUG
	//cudaMemcpy(mcPos_h, isspas.mcPos_d, nAtoms_h*isspas.nMC*sizeof(float4), cudaMemcpyDeviceToHost);
        //fprintf(isspas.denFile,"%d\n", nAtoms_h*isspas.nMC);
        //fprintf(isspas.denFile,"%d\n", nAtoms_h*isspas.nMC);
        //for (i=0;i<nAtoms_h*isspas.nMC; i++)
        //{
        //	fprintf(isspas.denFile,"C %10.6f %10.6f %10.6f %10.6f\n", mcPos_h[i].x, mcPos_h[i].y, mcPos_h[i].z, mcPos_h[i].w);
	//}
	// add to forces
	isspa_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize>>>(xyz_d, f_d, isspas.mcPos_d);
	// DEBUG
	//cudaMemcpy(mcPos_h, isspas.mcPos_d, nAtoms_h*isspas.nMC*sizeof(float4), cudaMemcpyDeviceToHost);
        //fprintf(isspas.forFile,"%d\n", nAtoms_h*isspas.nMC);
        //fprintf(isspas.forFile,"%d\n", nAtoms_h*isspas.nMC);
        //for (i=0;i<nAtoms_h*isspas.nMC; i++)
        //{
        //	fprintf(isspas.forFile,"C %10.6f %10.6f %10.6f %10.6f\n", mcPos_h[i].x, mcPos_h[i].y, mcPos_h[i].z, mcPos_h[i].w);
	//}
	//fclose(posFile);


	// finish timing
	cudaEventRecord(isspas.isspaStop);
	cudaEventSynchronize(isspas.isspaStop);
	cudaEventElapsedTime(&milliseconds, isspas.isspaStart, isspas.isspaStop);
	return milliseconds;

}

void isspa_grid_block(int nAtoms_h, int nPairs_h, float lbox_h, isspa& isspas) {

	int minGridSize;
	float hbox_h = lbox_h/2.0;
	float maxForce_h = MAXFORCE;

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
	cudaMemcpyToSymbol(nForceRs, &isspas.nForceRs, sizeof(int));
	cudaMemcpyToSymbol(nAtoms, &nAtoms_h, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &nPairs_h, sizeof(int));
	cudaMemcpyToSymbol(lbox, &lbox_h, sizeof(float));
	cudaMemcpyToSymbol(hbox, &hbox_h, sizeof(float));
	cudaMemcpyToSymbol(mu, &isspas.mu, sizeof(float));
	cudaMemcpyToSymbol(rho, &isspas.rho, sizeof(float));
	cudaMemcpyToSymbol(forceRparams, &isspas.forceRparams, sizeof(float2));
	cudaMemcpyToSymbol(nGRs, &isspas.nGRs, sizeof(int));
	cudaMemcpyToSymbol(gRparams, &isspas.gRparams, sizeof(float2));
	cudaMemcpyToSymbol(maxForce, &maxForce_h, sizeof(float));


}
