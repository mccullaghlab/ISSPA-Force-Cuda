#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
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

__global__  void isspa_MC_points_kernel(float4 *xyz, float4 *mcpos, curandState *state, const float* __restrict__ rmax, int *isspaTypes) {
        unsigned int MC = threadIdx.x + blockIdx.x*blockDim.x;
        int atom = blockIdx.x;
        int it;
        float r2;
        float rmax_l;
        float4 mcr;
        float4 mcpos_l;
        curandState_t threadState;

        // Determine which atom the MC point is being generated on
        it = __ldg(isspaTypes+atom);
        rmax_l = rmax[it];
        mcpos_l = __ldg(xyz+atom);
        threadState = state[MC];
        do {
                mcr.x = fmaf(2.0f,curand_uniform(&threadState),-1.0f);
                mcr.y = fmaf(2.0f,curand_uniform(&threadState),-1.0f);
                mcr.z = fmaf(2.0f,curand_uniform(&threadState),-1.0f);
                r2 = mcr.x*mcr.x + mcr.y*mcr.y + mcr.z*mcr.z;
        }
        while (r2 >= 0.99f);
        mcr *= rmax_l;
        mcpos_l += mcr;
        mcpos_l.w = 1.0f;
        mcpos[MC] = mcpos_l;
        state[MC] = threadState;
}



__global__ void isspa_force_kernel(float4 *xyz, const float* __restrict__ vtot, const float* __restrict__ rmax, int *isspaTypes, const float* __restrict__ forceTable, float4 *f, const float* __restrict__ gTable, const float* __restrict__ eTable, float4 *mcpos, int4 CalcsPerThread, float4 *isspaf) { 
        __shared__ float4 mcpos_s;
	__shared__ float4 enow_s;
	__shared__ float4 e0now_s;	
        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x;
	int atom;
        int atom2;
	int MC;
	int bin;
        int it;
	int jt;
        int i;
	//int MCind;
        float igo;
        float vtot_l;
        float rmax_l;
        float dist2, dist;
        float fracDist;
        float g1, g2;
        float e1, e2;
        float f1, f2;
        float etab;
        float r0;
	float dp1;
	float dp2;
	float dp3;
	float c1;
	float c2;
	float c3;
	float fs;
	float Rz;
	float cothE;
	float pdotr;
        float2 gRparams_l = gRparams;
        float2 eRparams_l = eRparams;
	float4 atom2_pos;
        float4 r;
        float4 mcpos_l;
        float4 enow_l;
        float4 e0now_l;
	float4 fi;
	float4 fj;
	// Zero out the field arrays
	enow_l.x = 0.0f;
	enow_l.y = 0.0f;
	enow_l.z = 0.0f;
	e0now_l.x = 0.0f;
	e0now_l.y = 0.0f;
	e0now_l.z = 0.0f;	  
	e0now_l.w = 0.0f;	  
	// Determine which atom the MC point is being generated on
	atom = int(__fdividef(index,(float) (nMC*CalcsPerThread.y)));
	// Determine the index of the  MC point being generated 
	MC = blockIdx.x;
	//MCind = int(MC - atom*nMC);
        
	// Get atom positions
        mcpos_l = __ldg(mcpos+MC);
	it = __ldg(isspaTypes+atom);
	rmax_l = rmax[it];
        vtot_l = vtot[it];
                
	// generate 3D MC pos based inside a sphere rnow based on MC point index		
	if (tid == 0) {
                mcpos_s = mcpos_l;
		enow_s = enow_l;
		e0now_s = e0now_l;		
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
			        e0now_l.w += 1.0f;
				// determine density bin of distance
				bin = int (__fdividef((dist-gRparams_l.x),gRparams_l.y)); 	
                                // make sure bin is in limits of density table
                                if (bin < 0) {
				        mcpos_l.w = 0.0f;
				} else if (bin < nGRs) {
				        // Push Density to MC point
					fracDist = __fdividef((dist - (gRparams_l.x+bin*gRparams_l.y)) , gRparams_l.y);   				        
					g1 = gTable[jt*nGRs+bin];
                                        g2 = gTable[jt*nGRs+bin+1];
					mcpos_l.w *= g1*(1.0f-fracDist)+g2*fracDist;
					// Push electric field to MC point
                                        fracDist = __fdividef((dist - (eRparams_l.x+bin*eRparams_l.y)) , eRparams_l.y);
                                        e1 = eTable[jt*nERs+bin];
                                        e2 = eTable[jt*nERs+bin+1];
                                        etab =  e1*(1.0f-fracDist)+e2*fracDist;
					enow_l += r*__fdividef(etab,dist);
				}      
                        } else {
                                e0now_l -= r*__fdividef(e0*atom2_pos.w,dist2*dist);
			}		
			enow_l -= r*__fdividef(e0*atom2_pos.w,dist2*dist);
		}		
	}

	mcpos_l.w = warpReduceMul(mcpos_l.w);	
	enow_l =  warpReduceSumTriple(enow_l);
	e0now_l =  warpReduceSumQuad(e0now_l);
        
	if ((threadIdx.x & (warpSize - 1)) == 0) {
	        atomicMul(&(mcpos_s.w), mcpos_l.w);
		atomicAdd(&(enow_s.x), enow_l.x);
		atomicAdd(&(enow_s.y), enow_l.y);
		atomicAdd(&(enow_s.z), enow_l.z);
		atomicAdd(&(e0now_s.x), e0now_l.x);
		atomicAdd(&(e0now_s.y), e0now_l.y);
		atomicAdd(&(e0now_s.z), e0now_l.z);
		atomicAdd(&(e0now_s.w), e0now_l.w);
        }
	__syncthreads();

        
	if (tid == 0) {
                printf("e0now_s: %f\n",e0now_s.w);
                // Convert enow into polarzation
                igo = __fdividef(vtot_l,e0now_s.w);
                //if (isinf(igo) == true) {
                //        printf("B mcpos: %f igo: %f e0now: %s\n",mcpos_s.w,igo,e0now_s.w);
                //}
                mcpos_s.w *= igo;	
                //if (isinf(mcpos_s.w) == true) {
                //        printf("B mcpos: %f igo: %f e0now: %s\n",mcpos_s.w,igo,e0now_s.w);
                //}
                r0 = norm3df(enow_s.x, enow_s.y, enow_s.z);
                enow_s /= r0;			
		enow_s.w = r0;
		e0now_s /= 3.0f;
		e0now_s.w = igo;
	}
	__syncthreads();

	mcpos_l = mcpos_s;
	enow_l = enow_s;
	e0now_l = e0now_s;

	for(i=0;i<CalcsPerThread.x;i++) {
	        // Determine which atom is generating the field at the MC point
	        atom2 = int(tid + i*CalcsPerThread.y);
		if (atom2 < nAtoms) {	
		        // Get atom positions
		        atom2_pos = __ldg(xyz+atom2);
		
			// Zero out the forces
			fi.x = 0.0f;
			fi.y = 0.0f;
			fi.z = 0.0f;
			fj.x = 0.0;
			fj.y = 0.0;
			fj.z = 0.0;
			
			// Get constants for atom
			jt = __ldg(isspaTypes+atom2);
			r = min_image(mcpos_l - atom2_pos,box.x,box.y);
			dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
			dist = sqrtf(dist2);			
			
			// Coulombic Force
			cothE=__fdividef(1.0f,tanhf(enow_l.w));
			c1=cothE-__fdividef(1.0f,enow_l.w);
			c2=1.0f-2.0f*__fdividef(c1,enow_l.w);
			c3=cothE-3.0f*__fdividef(c2,enow_l.w);
			
			Rz=__fdividef((enow_l.x*r.x+enow_l.y*r.y+enow_l.z*r.z),dist);
			dp1=3.0f*Rz;
			dp2=7.5f*Rz*Rz-1.5f;
			dp3=(17.50f*Rz*Rz-7.50f)*Rz;
                        fs = __fdividef(-atom2_pos.w*p0*c1*mcpos_l.w,dist2*dist);
                        fi += fs*(r*__fdividef(dp1,dist)-enow_l);
                        fj += fs*(r*__fdividef(dp1,dist)-enow_l);
                        // Calculate quadrapole term
                        fs = __fdividef(-atom2_pos.w*q0*(1.5f*c2-0.5f)*mcpos_l.w,dist2*dist2);
                        fi += fs*(r*__fdividef(dp2,dist)-dp1*enow_l);
                        fj += fs*(r*__fdividef(dp2,dist)-dp1*enow_l);
                        // Calculate octapole term
                        fs = __fdividef(-atom2_pos.w*o0*(2.5f*c3-1.5f*c1)*mcpos_l.w,dist2*dist2*dist);
                        fi += fs*(r*__fdividef(dp3,dist)-dp2*enow_l);
                        fj += fs*(r*__fdividef(dp3,dist)-dp2*enow_l);
			// Lennard-Jones Force 
			if (dist <= rmax_l) {
			        bin = int ( __fdividef((dist-forceRparams.x),forceRparams.y) + 0.5f);
				if (bin >= (nRs)) {
				        fs = 0.0f;
				} else {
				        //Lennard-Jones Force 
				        fracDist = __fdividef((dist-(forceRparams.x+bin*forceRparams.y)),forceRparams.y);
				        f1 = forceTable[it*nRs+bin];
				        f2 = forceTable[it*nRs+bin+1];
				        fs = (f1*(1.0-fracDist)+f2*fracDist)*mcpos_l.w;
                                        //fs = forceTable[jt*nRs+bin]*mcpos_l.w;
                                }
                                fi += r*__fdividef(-fs,dist);
				//fj += r*__fdividef(-fs,dist);
			} else {
			        // Constant Density Dielectric
			        fs=-atom2_pos.w*p0/dist2/dist;
				pdotr=__fdividef(3.0f*(e0now_l.x*r.x+e0now_l.y*r.y+e0now_l.z*r.z),dist2);
				fi += fs*(pdotr*r-e0now_l)*e0now_l.w;
				//fj += fs*(pdotr*r-e0now_l)*e0now_l.w;
			}						  	
			atomicAdd(&(f[atom2].x), fi.x);
			atomicAdd(&(f[atom2].y), fi.y);
			atomicAdd(&(f[atom2].z), fi.z);
			atomicAdd(&(isspaf[atom2].x), fj.x);
			atomicAdd(&(isspaf[atom2].y), fj.y);
			atomicAdd(&(isspaf[atom2].z), fj.z);
		}						
	}
}


/* C wrappers for kernels */
float isspa_force_cuda(float4 *xyz_d, float4 *f_d, float4 *isspaf_d, isspa& isspas, int nAtoms_h) {
//float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {

        float milliseconds;

	 // timing                                                                                                                
        cudaEventRecord(isspas.isspaStart);

	cudaProfilerStart();
        
        // compute position of each MC point
        isspa_MC_points_kernel<<<nAtoms_h,isspas.nMC >>>(xyz_d, isspas.mcpos_d, isspas.randStates_d, isspas.rmax_d, isspas.isspaTypes_d);
        // compute isspa forces for each atom 
	isspa_force_kernel<<<isspas.mcGridSize,isspas.mcBlockSize>>>(xyz_d,isspas.vtot_d,isspas.rmax_d,isspas.isspaTypes_d,isspas.isspaForceTable_d,f_d,isspas.isspaGTable_d,isspas.isspaETable_d,isspas.mcpos_d,isspas.mcCalcsPerThread,isspaf_d);

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
