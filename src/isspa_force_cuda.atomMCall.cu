
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

__global__ void isspa_mc_force_kernel(float4 *xyz, float4 *f, int nAtoms, float4 *mcDist, int *isspaTypes, float *forceTable, float2 *gTable, curandState *state) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float4 xyz_s[];
	int atom1;
	int atom2;
	float rnow;
	float x1, x2, r2;
	float4 atomPos;
	float4 mcPos;
	float4 rMC;
	float4 rMChat;
	float mcDens;
	float4 mcEmf;
	float4 fMC;
	float fLJ;
	float4 pol;
	float4 polHat;
	float magPol;
	float magEmf;
	float4 r;
	int it, jt;
	int i;
	int bin;
	float2 gnow;
	float dist2, dist;
	float mu_l = mu;
	float hbox_l = hbox;
	float lbox_l = lbox;
	float4 mcDist_l;
	float2 forceRparams_l = forceRparams;
	int localnForceRs = nForceRs;
	int localnAtoms = nAtoms;
	int localnMC = nMC;
	float2 gRparams_l = gRparams;
	int localnGRs=nGRs;
	curandState_t threadState;

	// copy atom positions to shared memory
	for (i=t;i<localnAtoms;i+=blockDim.x) {
		xyz_s[i] = __ldg(xyz+i);
	}
	__syncthreads();
	// move on
	if (index < localnAtoms*localnMC)
	{

		// random number state - store in temporary variable
		threadState = state[index];
		// get atom number of interest
		atom1 = int(index/(float) localnMC);
		atomPos = xyz_s[atom1];
		// isspa type
		it = __ldg(isspaTypes+atom1);
		mcDist_l = __ldg(mcDist + it);
		// select uniform number between rmin and rmax of MC dist
		rnow = mcDist_l.x + mcDist_l.z * curand_uniform(&threadState);
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
		rMC.x =  rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		rMC.y = rnow*x1*r2;
		rMC.z = rnow*x2*r2;
		// save force between MC point and atom of interest
		bin = int ( (rnow-forceRparams_l.x)/forceRparams_l.y );
		if (bin >= (localnForceRs-1)) {
			fLJ = 0.0;
		} else {
			// linearly interpolate between two force bins
			//fracDist = (dist - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
			//f1 = __ldg(forceTable+it*nForceRs+bin);
			//f2 = __ldg(forceTable+it*nForceRs+bin+1);
			//fLJ = f1*(1.0-fracDist)+f2*fracDist;
			fLJ = __ldg(forceTable+it*localnForceRs+bin);
		}
		// add atom position to mc point position
		mcPos = rMC + atomPos;
		// initialize density to N/P(r) = rho*4*pi*(r2-r1)*r^2/nMC
		mcDens = mcDist_l.w*rnow*rnow;
		mcEmf.x = 0.0;  // initiate E_{MF}
		mcEmf.y = 0.0;
		mcEmf.z = 0.0;
		// save random state in global
		state[index] = threadState;

		// loop through all neighboring atom to compute g and E_{MF} at MC point
		for (atom2=0;atom2<nAtoms;atom2++) {
			r = min_image(mcPos - xyz_s[atom2],lbox_l,hbox_l);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			dist = sqrtf(dist2);
			// determine density bin of distance
			bin = int ( (dist-gRparams_l.x)/gRparams_l.y );
			// make sure bin is in limits of density table
			if (bin >= (localnGRs-1)) {
			} else if (bin <= 0) {
				mcDens = 0.0;
				break;
			} else {
				// get ISSPA atom type of atom of interest
				jt = __ldg(isspaTypes + atom2);
				// linearly interpolate between two density bins
				//fracDist = (dist - (gRparams.x+bin*gRparams.y)) / gRparams.y;
				//g1 = __ldg(gTable+jt*nGRs+bin);
				//g2 = __ldg(gTable+jt*nGRs+bin+1);
				//gnow = g1*(1.0-fracDist)+g2*fracDist;
				gnow = __ldg(gTable + jt*localnGRs+bin);
				mcDens *= gnow.x; // Density
				mcEmf += gnow.y/dist * r; // Mean field
			}

			if (mcDens < THRESH) {
				break;
			}

		}
		
		if (mcDens > THRESH) {	
			// calculate forces
			// LJ force
			rMChat = rMC/rnow;
			fMC = -rMChat*fLJ;
			// Coulomb dipole force
			magEmf = sqrtf(mcEmf.x*mcEmf.x + mcEmf.y*mcEmf.y + mcEmf.z*mcEmf.z);
			pol  = (1.0/tanhf(magEmf) - 1.0/magEmf)/magEmf  * mcEmf ;
			magPol = sqrtf(pol.x*pol.x + pol.y*pol.y + pol.z*pol.z);
			polHat = pol/magPol;
			fMC += mu_l*atomPos.w*magPol*(3.0*(polHat.x*rMChat.x + polHat.y*rMChat.y + polHat.z*rMChat.z)*rMChat - polHat);
			// weight force by density
			fMC *= mcDens;
			// add force to atom
			atomicAdd(&(f[atom1].x), fMC.x);
			atomicAdd(&(f[atom1].y), fMC.y);
			atomicAdd(&(f[atom1].z), fMC.z);
		}
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
	isspa_mc_force_kernel<<<isspas.mcGridSize, isspas.mcBlockSize,nAtoms_h*sizeof(float4)>>>(xyz_d, f_d, nAtoms_h, isspas.mcDist_d, isspas.isspaTypes_d, isspas.isspaForceTable_d, isspas.isspaGTable_d, isspas.randStates_d);
	// compute density at each mc point
	// DEBUG
	//cudaMemcpy(mcPos_h, isspas.mcPos_d, nAtoms_h*isspas.nMC*sizeof(float4), cudaMemcpyDeviceToHost);
        //fprintf(isspas.denFile,"%d\n", nAtoms_h*isspas.nMC);
        //fprintf(isspas.denFile,"%d\n", nAtoms_h*isspas.nMC);
        //for (i=0;i<nAtoms_h*isspas.nMC; i++)
        //{
        //	fprintf(isspas.denFile,"C %10.6f %10.6f %10.6f %10.6f\n", mcPos_h[i].x, mcPos_h[i].y, mcPos_h[i].z, mcPos_h[i].w);
	//}

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
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.mcBlockSize, isspa_mc_force_kernel, 0, nAtoms_h*isspas.nMC);
    	// Round up according to array size
    	isspas.mcGridSize = (nAtoms_h*isspas.nMC + isspas.mcBlockSize - 1) / isspas.mcBlockSize;

	// determine gridSize and blockSize
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &isspas.gBlockSize, isspa_density_kernel, 0, nPairs_h*isspas.nMC);
    	// Round up according to array size
    	//isspas.gGridSize = (nPairs_h*isspas.nMC + isspas.gBlockSize - 1) / isspas.gBlockSize;

	// set constant memory
	cudaMemcpyToSymbol(nMC, &isspas.nMC, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &isspas.nTypes, sizeof(int));
	cudaMemcpyToSymbol(nForceRs, &isspas.nForceRs, sizeof(int));
	cudaMemcpyToSymbol(nAtoms, &nAtoms_h, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &nPairs_h, sizeof(int));
	cudaMemcpyToSymbol(mu, &isspas.mu, sizeof(float));
	cudaMemcpyToSymbol(lbox, &lbox_h, sizeof(float));
	cudaMemcpyToSymbol(hbox, &hbox_h, sizeof(float));
	cudaMemcpyToSymbol(forceRparams, &isspas.forceRparams, sizeof(float2));
	cudaMemcpyToSymbol(nGRs, &isspas.nGRs, sizeof(int));
	cudaMemcpyToSymbol(gRparams, &isspas.gRparams, sizeof(float2));
	cudaMemcpyToSymbol(maxForce, &maxForce_h, sizeof(float));


}
