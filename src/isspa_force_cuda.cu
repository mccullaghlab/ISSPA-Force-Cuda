#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_vector_routines.h"
#include "isspa_class.h"
#include "isspa_force_cuda.h"
#include "constants.h"
#include "cuda_profiler_api.h"

using namespace std;

// constants
__constant__ int nTypes;
__constant__ int nMC;
__constant__ int nRs;
__constant__ int nGRs;
__constant__ int nERs;
__constant__ int nAtoms;
__constant__ int nPairs;
__constant__ float2 box;
__constant__ float2 forceRparams;
__constant__ float2 gRparams;
__constant__ float2 eRparams;

// device functions

// CUDA Kernels

__device__ float atomicMul(float* address, float val) { 
  unsigned int* address_as_u = (unsigned int*)address; 
  unsigned int old = *address_as_u, assumed; 
        do { 
	        assumed = old; 
		old = atomicCAS(address_as_u, assumed, __float_as_uint(val * __uint_as_float(assumed))); 
	} while (assumed != old); return __uint_as_float(old);
}

__inline__ __device__ float warpReduceMul(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val *= __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float4 warpReduceSumQuad(float4 val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
    val.z += __shfl_down(val.z, offset);
    val.w += __shfl_down(val.w, offset);
  }
  return val; 
}

__inline__ __device__
float4 warpReduceSumTriple(float4 val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
    val.z += __shfl_down(val.z, offset);
   }
  return val; 
}

__inline__ __device__
float4 createMonteCarloPoint(float rmax, curandState_t threadState, int MC) {
  float r2;
  float4 mcr;
  do {
    mcr.x = (2.0f * curand_uniform(&threadState) - 1.0f);
    mcr.y = (2.0f * curand_uniform(&threadState) - 1.0f);
    mcr.z = (2.0f * curand_uniform(&threadState) - 1.0f);
    r2 = mcr.x*mcr.x + mcr.y*mcr.y + mcr.z*mcr.z;
  }
  while (r2 >= 1.0f);
  mcr *= rmax;
  
  return mcr; 
}

__global__ void isspa_field_kernel(float4 *xyz, float *vtot, float *rmax, int *isspaTypes, float *gTable, float *eTable, curandState *state, float4 *enow, float4 *e0now, float4 *mcpos, int4 CalcsPerThread) { 
        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x;
	int atom;
        int atom2;
	int MC;
	int bin;
        int it;
	int jt;
        int i;
	int MCind;
        float igo;
        float vtot_l;
        float rmax_l;
        float dist2, dist;
        float fracDist;
        float g1, g2;
        float e1, e2;
        float etab;
        float r2;
        float r0;
        float2 gRparams_l = gRparams;
        float2 eRparams_l = eRparams;
	float4 atom_pos;
	float4 atom2_pos;
        float4 mcr;
        float4 r;
        float4 mcpos_l;
        float4 enow_l;
        float4 e0now_l;
        curandState_t threadState;
	enow_l.x = 0.0;
	enow_l.y = 0.0;
	enow_l.z = 0.0;
	e0now_l.x = 0.0;
	e0now_l.y = 0.0;
	e0now_l.z = 0.0;	  
	e0now_l.w = 0.0;	  
	// Determine which atom the MC point is being generated on
	atom = int(index/(float) (nMC*CalcsPerThread.y));
	// Determine the index of the  MC point being generated 
	MC = blockIdx.x;
	MCind = int(MC - atom*nMC);
	// Get atom positions
	atom_pos = __ldg(xyz+atom);
	mcpos_l = atom_pos;
	it = __ldg(isspaTypes+atom);
	rmax_l = __ldg(rmax+it);
	vtot_l = __ldg(vtot+it);
	// generate 3D MC pos based inside a sphere rnow based on MC point index		
	threadState = state[MC];
	mcr = createMonteCarloPoint(rmax_l, threadState, MCind);
	mcpos_l += mcr;
	mcpos_l.w = 1.0;
	if (tid == 0) {
	  mcpos[MC] = mcpos_l;
	}
	for(i=0;i<CalcsPerThread.x;i++) {
	        // Determine which atom is generating the field at the MC point
	        atom2 = int(tid + i*CalcsPerThread.y);
		if (atom2 < nAtoms) {	
		        // Get atom positions
		        atom2_pos = __ldg(xyz+atom2);
			// Get constants for atom
			jt = __ldg(isspaTypes+atom2);
			r = min_image(mcpos_l - atom2_pos,box.x,box.y);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			dist = sqrtf(dist2);			
			if (dist <= rmax_l) {
			        e0now_l.w += 1;
				// determine density bin of distance
				bin = int ( (dist-gRparams_l.x)/gRparams_l.y ); 	
				// make sure bin is in limits of density table
				if (bin < 0) {
				        mcpos_l.w = 0.0;
				} else if (bin < nGRs) {
				        // Push Density to MC point
					fracDist = (dist - (gRparams_l.x+bin*gRparams_l.y)) / gRparams_l.y;   				        
					g1 = __ldg(gTable+jt*nGRs+bin);
					g2 = __ldg(gTable+jt*nGRs+bin+1);
					mcpos_l.w *= g1*(1.0-fracDist)+g2*fracDist;
					// Push electric field to MC point
					fracDist = (dist - (eRparams_l.x+bin*eRparams_l.y)) / eRparams_l.y;
					e1 = __ldg(eTable+jt*nERs+bin);
					e2 = __ldg(eTable+jt*nERs+bin+1);					
					etab =  e1*(1.0-fracDist)+e2*fracDist;
					enow_l += r/dist*etab;
				}      
			} else {
			        e0now_l -= e0*atom2_pos.w*r/dist2/dist;
			}		
			enow_l -= e0*atom2_pos.w*r/dist2/dist;
		}		
	}
	
	mcpos_l.w = warpReduceMul(mcpos_l.w);	
	enow_l =  warpReduceSumTriple(enow_l);
	e0now_l =  warpReduceSumQuad(e0now_l);
	
	if ((threadIdx.x & (warpSize - 1)) == 0) {
	        atomicMul(&(mcpos[MC].w), mcpos_l.w);
		atomicAdd(&(enow[MC].x), enow_l.x);
		atomicAdd(&(enow[MC].y), enow_l.y);
		atomicAdd(&(enow[MC].z), enow_l.z);
		atomicAdd(&(e0now[MC].x), e0now_l.x);
		atomicAdd(&(e0now[MC].y), e0now_l.y);
		atomicAdd(&(e0now[MC].z), e0now_l.z);
		atomicAdd(&(e0now[MC].w), e0now_l.w);
	}
	
	__syncthreads();
	  
	  
	if (tid == 0) {
		// Convert enow into polarzation
	        igo = vtot_l/e0now[MC].w;
		mcpos[MC].w *= igo;	
		r2 = enow[MC].x*enow[MC].x+enow[MC].y*enow[MC].y+enow[MC].z*enow[MC].z;
		r0 = sqrtf(r2);
		enow[MC] /= r0;			
		enow[MC].w = r0;
		e0now[MC] /= 3.0;
		e0now[MC].w = igo;
	}
}

__global__ void isspa_force_kernel(float4 *xyz, float *vtot, float *rmax, int *isspaTypes, float *forceTable, float4 *f, curandState *state,  float4 *enow, float4 *e0now, float4 *mcpos, float4 *isspaf) {
        int bin;
        int jt;
        float fs;
	//float f1, f2;
        float rmax_l;
        float dist2, dist;
        float pdotr;
	float cothE;
	float c1,c2,c3;
	float dp1,dp2,dp3;
	float Rz;
	float4 xyz_l;
        float4 r;
        float4 fi;
        float4 fj;
	float4 mcpos_l;
	float4 enow_l;
	float4 e0now_l;

	// Determine the atom for which the force is being summed on
	int atom = int(blockIdx.x/(float) (nAtoms));
	// Determine which atom to grab the MC points from
	int atom2 = (blockIdx.x - atom*nAtoms);
	// Determine the index of the MC point for atom2
	int MC = atom2*nMC + threadIdx.x;

	// Zero out the forces
	fi.x = 0.0;
	fi.y = 0.0;
	fi.z = 0.0;
	fj.x = 0.0;
	fj.y = 0.0;
	fj.z = 0.0;

	if (threadIdx.x < nMC) { 		
	        // Load in position, atom type, and rmax of atom
		xyz_l = __ldg(xyz+atom);
		jt = __ldg(isspaTypes + atom);
		rmax_l = __ldg(rmax+jt);
		
		// Load in field data for the MC point	
		mcpos_l = __ldg(mcpos+MC);
		enow_l = __ldg(enow+MC);
		e0now_l = __ldg(e0now+MC);

		// Calculate the distance between the MC point and atom1
		r = min_image(mcpos_l - xyz_l,box.x,box.y);
		dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
		dist = sqrtf(dist2);
		
		// Coulombic Force
		cothE=1.0/tanh(enow_l.w);
		c1=cothE-1.0/enow_l.w;
		c2=1.0-2.0*c1/enow_l.w;
		c3=cothE-3.0*c2/enow_l.w;
		
		Rz=(enow_l.x*r.x+enow_l.y*r.y+enow_l.z*r.z)/dist;
		dp1=3.0*Rz;
		dp2=7.5*Rz*Rz-1.5;
		dp3=(17.50*Rz*Rz-7.50)*Rz;
		fs =-xyz_l.w*p0*c1/dist2/dist*mcpos_l.w;
		fi += fs*(dp1*r/dist-enow_l);
		//fj += fs*(dp1*r/dist-enow_l);
		fs =-xyz_l.w*q0*(1.5*c2-0.5)/dist2/dist2*mcpos_l.w;
		fi += fs*(dp2*r/dist-dp1*enow_l);
		//fj += fs*(dp2*r/dist-dp1*enow_l);
		fs =-xyz_l.w*o0*(2.5*c3-1.5*c1)/dist2/dist2/dist*mcpos_l.w;
		fi += fs*(dp3*r/dist-dp2*enow_l);
		//fj += fs*(dp3*r/dist-dp2*enow_l);

		// Lennard-Jones Force 
		if (dist <= rmax_l) {
		        bin = int ( (dist-forceRparams.x)/forceRparams.y + 0.5f);
			if (bin >= (nRs)) {
		              fs = 0.0;
			} else {
			        //Lennard-Jones Force 
				//fracDist = (dist - (forceRparams.x+bin*forceRparams.y)) / forceRparams.y;
				//f1 = __ldg(forceTable+it*nRs+bin);
				//f2 = __ldg(forceTable+it*nRs+bin+1);
				//fs = (f1*(1.0-fracDist)+f2*fracDist)*mcpos.w;
			        fs = __ldg(forceTable + jt*nRs+bin)*mcpos_l.w;
			}
			      fi += -fs*r/dist;
			      fj += -fs*r/dist;
		} else {
		        // Constant Density Dielectric
		        fs=-xyz_l.w*p0/dist2/dist;
			pdotr=3.0*(e0now_l.x*r.x+e0now_l.y*r.y+e0now_l.z*r.z)/dist2;
			fi += fs*(pdotr*r-e0now_l)*e0now_l.w;
			//fj += fs*(pdotr*r-e0now_l)*e0now_l.w;			
		}	
	}
	
	fi =  warpReduceSumTriple(fi);
	fj =  warpReduceSumTriple(fj);
	  	
	if ((threadIdx.x & (warpSize - 1)) == 0) {
	        atomicAdd(&(f[atom].x), fi.x);
		atomicAdd(&(f[atom].y), fi.y);
		atomicAdd(&(f[atom].z), fi.z);
		atomicAdd(&(isspaf[atom].x), fj.x);
		atomicAdd(&(isspaf[atom].y), fj.y);
		atomicAdd(&(isspaf[atom].z), fj.z);
	}	
}

/* C wrappers for kernels */

float isspa_force_cuda(float4 *xyz_d, float4 *f_d, float4 *isspaf_d, isspa& isspas, int nAtoms_h) {
//float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {

        float milliseconds;

	 // timing                                                                                                                
        cudaEventRecord(isspas.isspaStart);

	cudaProfilerStart();
	
        // compute densities and mean electric field value for each MC point
	isspa_field_kernel<<<isspas.mcGridSize, isspas.mcBlockSize, isspas.mcCalcsPerThread.z*sizeof(float)>>>(xyz_d,isspas.vtot_d,isspas.rmax_d,isspas.isspaTypes_d,isspas.isspaGTable_d,isspas.isspaETable_d,isspas.randStates_d,isspas.enow_d,isspas.e0now_d,isspas.mcpos_d,isspas.mcCalcsPerThread);
       
	// compute forces for each atom
	isspa_force_kernel<<<isspas.fGridSize, isspas.fBlockSize>>>(xyz_d,isspas.vtot_d,isspas.rmax_d,isspas.isspaTypes_d,isspas.isspaForceTable_d,f_d,isspas.randStates_d,isspas.enow_d,isspas.e0now_d,isspas.mcpos_d,isspaf_d);

	cudaDeviceSynchronize();
	cudaProfilerStop();
	  
        // finish timing
	cudaEventRecord(isspas.isspaStop);
	cudaEventSynchronize(isspas.isspaStop);
	cudaEventElapsedTime(&milliseconds, isspas.isspaStart, isspas.isspaStop);
	return milliseconds;
}

void isspa_grid_block(int nAtoms_h, int nPairs_h, float lbox_h, isspa& isspas) {
  
        float2 box_h;
	int maxThreadsPerBlock = 1024;
	int temp;

	// determine gridSize and blockSize for field kernel	
	if (nAtoms_h <= maxThreadsPerBlock) {
	        temp = int(ceil(nAtoms_h /(float) 32.0));
		isspas.mcCalcsPerThread.y = 32*temp;
		isspas.mcBlockSize = isspas.mcCalcsPerThread.y;		
	} else {
	        isspas.mcBlockSize = maxThreadsPerBlock;	  
		isspas.mcCalcsPerThread.y = maxThreadsPerBlock;	  	
	}
	isspas.mcCalcsPerThread.x = int(ceil(nAtoms_h/ (float) isspas.mcBlockSize));	  	
	isspas.mcGridSize = nAtoms_h*isspas.nMC;	
	printf("Number of field kernel blocks: %d \n", isspas.mcGridSize);
	printf("Number of field kernel threads per block: %d \n", isspas.mcBlockSize);
	printf("Number of field kernel ISSPA MC-atom pair calculations per thread: %d \n", isspas.mcCalcsPerThread.x);
	
        // determine gridSize and blockSize for force kernel
	isspas.fGridSize = nAtoms_h*nAtoms_h;
	temp = int(ceil(isspas.nMC /(float) 32));
	isspas.fBlockSize = temp*32;		   
	printf("Number of force kernel blocks: %d \n", isspas.fGridSize);
	printf("Number of force kernel threads per block: %d \n", isspas.fBlockSize);
	
	// fill box with box and half box length
	box_h.x = lbox_h;
	box_h.y = lbox_h/2.0;
	
	// set constant memory
	cudaMemcpyToSymbol(nMC, &isspas.nMC, sizeof(int));
	cudaMemcpyToSymbol(nTypes, &isspas.nTypes, sizeof(int));
	cudaMemcpyToSymbol(nRs, &isspas.nRs, sizeof(int));
	cudaMemcpyToSymbol(nGRs, &isspas.nGRs, sizeof(int));
	cudaMemcpyToSymbol(nERs, &isspas.nERs, sizeof(int));
	cudaMemcpyToSymbol(nAtoms, &nAtoms_h, sizeof(int));
	cudaMemcpyToSymbol(nPairs, &nPairs_h, sizeof(int));
	cudaMemcpyToSymbol(box, &box_h, sizeof(float2));
	cudaMemcpyToSymbol(forceRparams, &isspas.forceRparams, sizeof(float2));
	cudaMemcpyToSymbol(gRparams, &isspas.gRparams, sizeof(float2));	
	cudaMemcpyToSymbol(eRparams, &isspas.eRparams, sizeof(float2));	
}
