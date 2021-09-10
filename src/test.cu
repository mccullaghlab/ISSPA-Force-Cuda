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

// atomic multiply
__device__ float atomicMul(float* address, float val) { 
        unsigned int* address_as_u = (unsigned int*)address; 
        unsigned int old = *address_as_u, assumed; 
        do { 
	        assumed = old; 
		old = atomicCAS(address_as_u, assumed, __float_as_uint(val * __uint_as_float(assumed))); 
	} while (assumed != old); return __uint_as_float(old);
}

// warp reduce a float using multiplication
__inline__ __device__ float warpReduceMul(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
          val *= __shfl_down(val, offset);
  return val;
}

// warp reduce a float4
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

// warp reduce a float4 but only the first three values
__inline__ __device__
float4 warpReduceSumTriple(float4 val) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val.x += __shfl_down(val.x, offset);
                val.y += __shfl_down(val.y, offset);
                val.z += __shfl_down(val.z, offset);
        }
        return val; 
}

// kernel to generate MC points around each atom
__global__  void isspa_MC_points_kernel(float4 *xyz, float4 *mcpos, curandState *state, const float* __restrict__ rmax, int *isspaTypes) {
        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;    
        int atom;
	int it;
	float r2;
	float rmax_l;
	float4 mcr;
	float4 mcpos_l;
	curandState_t threadState;
        
        atom = int(double(index)/double(nMC));
        if (atom < nAtoms) {
                // load atom paramters
                it = __ldg(isspaTypes+atom);
                rmax_l = rmax[it];
                mcpos_l = __ldg(xyz+atom);
                // initialize the random state
                threadState = state[index];
                // generate point in constant density sphere	
                do {
                        mcr.x = fmaf(2.0f,curand_uniform(&threadState),-1.0f);
                        mcr.y = fmaf(2.0f,curand_uniform(&threadState),-1.0f);
                        mcr.z = fmaf(2.0f,curand_uniform(&threadState),-1.0f);
                        r2 = mcr.x*mcr.x + mcr.y*mcr.y + mcr.z*mcr.z;
                } while (r2 >= 0.99f);
                // expand sphere and translate by atom position
                mcr *= rmax_l;
                mcpos_l += mcr;
                // initialize density at MC point to 1
                mcpos_l.w = 1.0f;
                // save MC point and random state back to global memory
                mcpos[index] = mcpos_l;
                state[index] = threadState;
        }
}

// kernel to compute density and mean field at each MC point
__global__ void
__launch_bounds__(1024, 3)
isspa_field_kernel(float4 *xyz, const float* __restrict__ rmax, const float* __restrict__ vtot, int *isspaTypes, const float* __restrict__  gTable, const float* __restrict__  eTable, float4 *enow, float4 *e0now, float4 *mcpos) { 
	int atom;
        int atom2;
	int MC;
	int bin;
        int it;
	int jt;
	float r0;
	float igo;
        float rmax_l;
	float vtot_l;
        float dist2, dist;
        float fracDist;
        float etab;
        float4 r;
        float4 mcpos_l;
        float4 enow_l;
        float4 e0now_l;

        // Determine the index of the thread within the warp
        unsigned int idx = threadIdx.x - int(threadIdx.x/32)*32;
        // Load in atom data in shared memory
        extern __shared__ float4 xyz_s[];
        for (unsigned int offset  = 0; offset < nAtoms; offset += blockDim.x) {
                if (offset + threadIdx.x < nAtoms) {
                        xyz_s[offset + threadIdx.x] = xyz[offset + threadIdx.x];
                }
        }
        __syncthreads();

        // Determine the atom and MC indicies for the current warp
        atom = int(blockIdx.x);
        // Need to load the isspatypes in shared memory
        // Read in parameters for the atom
        it = __ldg(isspaTypes+atom);
        rmax_l = rmax[it];                                
        vtot_l = vtot[it];
        for (unsigned int MCoffset = 0; MCoffset < nMC; MCoffset += 32) {
                MC = int(threadIdx.x/32.0f + MCoffset + atom*nMC);
                mcpos_l = __ldg(mcpos+MC);
                // If nMC is not a multiple of 32 then there will extra warps initiated 
                bool active = true;
                if (MC-atom*nMC >= nMC) {
                        active = false;
                }
                // zero the local variables that will be reduced
                enow_l.x = enow_l.y = enow_l.z = enow_l.w = 0.0f;
                e0now_l.x = e0now_l.y = e0now_l.z = e0now_l.w = 0.0f;
                if (active) {
                        // identify the atom we are about to handle
                        for (unsigned int offset = 0; offset < nAtoms; offset += 32) {
                                atom2 = offset + idx;
                                if (atom2 < nAtoms) {
                                        // Get constants for atom
                                        jt = __ldg(isspaTypes+atom2);
                                        r = min_image(mcpos_l - xyz_s[atom2],box.x,box.y);
                                        dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
                                        dist = sqrtf(dist2);
                                        if (dist <= rmax_l) {
                                                e0now_l.w += 1.0f;
                                                // determine density bin of distance
                                                bin = int(__fdividef(dist-gRparams.x,gRparams.y)); 	
                                                // make sure bin is in limits of density table
                                                if (bin < 0) {
                                                        mcpos_l.w = 0.0f;
                                                } else if (bin < nGRs-1) {
                                                        // Push Density to MC point
                                                        fracDist = __fdividef((dist - (gRparams.x+bin*gRparams.y)),gRparams.y);
                                                        mcpos_l.w *= fmaf( gTable[jt*nGRs+bin+1],fracDist, gTable[jt*nGRs+bin]*(1.0f-fracDist));
                                                        // Push mean field to MC point
                                                        fracDist = __fdividef((dist - (eRparams.x+bin*eRparams.y)),eRparams.y);
                                                        etab =  fmaf(eTable[jt*nERs+bin+1],fracDist,eTable[jt*nERs+bin]*(1.0f-fracDist));
                                                        enow_l += r*__fdividef(etab,dist);
                                                }      
                                        } else {
                                                e0now_l -= r*__fdividef(e0*xyz_s[atom2].w,dist2*dist);
                                        }				
                                        enow_l -= r*__fdividef(e0*xyz_s[atom2].w,dist2*dist);        
                                }
                        }
                        // Warp reduce the fields
                        mcpos_l.w = warpReduceMul(mcpos_l.w);	
                        enow_l =  warpReduceSumTriple(enow_l);
                        e0now_l =  warpReduceSumQuad(e0now_l);
                        // Add the fields to the global variable
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
                }
        }
        __syncthreads();
        
        // finsih calculating density, enow, e0now
        for (unsigned int MCoffset = 0; MCoffset < nMC; MCoffset += 32) {
                MC = int(threadIdx.x/32.0f + MCoffset + atom*nMC);                        
                if ( MC-atom*nMC < nMC ) {
                        if ((threadIdx.x & (warpSize - 1)) == 0) {
                                // Finish calculating density                                                                                              
                                igo = __fdividef(vtot_l,e0now[MC].w);
                                mcpos[MC].w *= igo;
                                // Convert enow into polarization                                                                                           
                                r0 = norm3df(enow[MC].x, enow[MC].y, enow[MC].z);
                                enow[MC].x = __fdividef(enow[MC].x,r0);
                                enow[MC].y = __fdividef(enow[MC].y,r0);
                                enow[MC].z = __fdividef(enow[MC].z,r0);
                                enow[MC].w = r0;
                                e0now[MC].x = __fdividef(e0now[MC].x,3.0f);
                                e0now[MC].y = __fdividef(e0now[MC].y,3.0f);
                                e0now[MC].z = __fdividef(e0now[MC].z,3.0f);
                                e0now[MC].w = igo;
                        }
                }
        }
}	

__global__ void
__launch_bounds__(1024, 2)
isspa_force_kernel(float4 *xyz, const float* __restrict__ rmax, int *isspaTypes, const float* __restrict__ forceTable, float4 *f, float4 *enow, float4 *e0now, float4 *mcpos, float4 *isspaf) {
	int bin;
        int jt;
	int MC;
	int atom;
        int atom2;
        float nBlocks_per_atom;
	float fs;
        float rmax_l;
        float dist2, dist;
	float cothE;
	float c1,c2,c3;
	float dp1,dp2,dp3;
	float Rz;
        float fracDist;
        float4 r;
        float4 mcpos_l;
        float4 enow_l;
        float4 e0now_l;
        float4 fi;
        //float4 fj;

        // Determine the index of the thread within the warp
        unsigned int idx = threadIdx.x - int(threadIdx.x/32.0f)*32;
        // Load in atom data in shared memory
        extern __shared__ float4 xyz_s[];
        for (unsigned int offset  = 0; offset < nAtoms; offset += blockDim.x) {
                if (offset + threadIdx.x < nAtoms) {
                        xyz_s[offset + threadIdx.x] = xyz[offset + threadIdx.x];
                }
        }
        __syncthreads();

        nBlocks_per_atom = ceil(blockDim.x/32.0f);
        atom = int(blockIdx.x / nBlocks_per_atom);                        

        // identify the atom from which the MC points are on
        atom2 = int(threadIdx.x/32.0f + (blockIdx.x - atom*nBlocks_per_atom)*32);
        // Load in position, atom type, and rmax of atom
        jt = __ldg(isspaTypes + atom2);
        rmax_l = rmax[jt];                
        bool active = true;
        if (atom2 >= nAtoms) {
                active = false;
        }                
        // Zero out the forces
        fi.x = fi.y = fi.z = 0.0f;
        //fj.x = fj.y = fj.z = 0.0f;                        
        if (active) {
                // loop over the MC points for each atom 
                for (unsigned int offset = 0; offset < nMC; offset += 32) {
                        // Determine the MC points from atom2
                        MC = idx + offset + atom*nMC;
                        if (MC-atom*nMC < nMC) {
                                // Load in field data for the MC point
                                mcpos_l = __ldg(mcpos+MC);
                                enow_l = __ldg(enow+MC);
                                e0now_l = __ldg(e0now+MC);
                                // Calculate the distance between the MC point and atom1
                                r = min_image(mcpos_l - xyz_s[atom2],box.x,box.y);
                                dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
                                dist = sqrtf(dist2);                
                                // Coulombic Force
                                cothE=__fdividef(1.0f,tanhf(enow_l.w));
                                c1=cothE-__fdividef(1.0f,enow_l.w);
                                c2=1.0f-2.0f*__fdividef(c1,enow_l.w);
                                c3=cothE-3.0f*__fdividef(c2,enow_l.w);                
                                Rz=__fdividef(enow_l.x*r.x+enow_l.y*r.y+enow_l.z*r.z,dist);
                                dp1=3.0f*Rz;
                                dp2=7.5f*Rz*Rz-1.5f;
                                dp3=(17.50f*Rz*Rz-7.50f)*Rz;                
                                // Calculate dipole term
                                fs = __fdividef(-xyz_s[atom2].w*p0*c1*mcpos_l.w,dist2*dist);
                                fi += fs*(r*__fdividef(dp1,dist)-enow_l);
                                //fj += fs*(r*__fdividef(dp1,dist)-enow_l);
                                // Calculate quadrapole term
                                fs = __fdividef(-xyz_s[atom2].w*q0*(1.5f*c2-0.5f)*mcpos_l.w,dist2*dist2);
                                fi += fs*(r*__fdividef(dp2,dist)-dp1*enow_l);
                                //fj += fs*(r*__fdividef(dp2,dist)-dp1*enow_l);
                                // Calculate octapole term
                                fs = __fdividef(-xyz_s[atom2].w*o0*(2.5f*c3-1.5f*c1)*mcpos_l.w,dist2*dist2*dist);
                                fi += fs*(r*__fdividef(dp3,dist)-dp2*enow_l);
                                //fj += fs*(r*__fdividef(dp3,dist)-dp2*enow_l);
                                // Lennard-Jones Force  
                                if (dist <= rmax_l) {
                                        bin = int ( __fdividef(dist-forceRparams.x,forceRparams.y) + 0.5f);
                                        if (bin >= (nRs)) {
                                                fs = 0.0f;
                                        } else {
                                                //Lennard-Jones Force
                                                fracDist = __fdividef((dist-(forceRparams.x+bin*forceRparams.y)),forceRparams.y);
                                                fs =  fmaf(forceTable[jt*nRs+bin+1],fracDist, forceTable[jt*nRs+bin]*(1.0f-fracDist))*mcpos_l.w;
                                        }
                                        fi += r*__fdividef(-fs,dist);
                                        //fj += r*__fdividef(-fs,dist);
                                } else {
                                        // Constant Density Dielectric
                                        fs=__fdividef(-xyz_s[atom2].w*p0,dist2*dist);
                                        fi -= fs*(__fdividef(3.0f*(e0now_l.x*r.x+e0now_l.y*r.y+e0now_l.z*r.z),dist2)*r-e0now_l)*e0now_l.w;
                                        //fj -= fs*(pdotr*r-e0now_l)*e0now_l.w;
                                }	
                                }
                }
                // Warp reduce the forces
                fi =  warpReduceSumTriple(fi);
                //fj =  warpReduceSumTriple(fj);
                // Add the force to the global force
                if ((threadIdx.x & (warpSize - 1)) == 0) {
                        atomicAdd(&(f[atom2].x), fi.x);
                        atomicAdd(&(f[atom2].y), fi.y);
                        atomicAdd(&(f[atom2].z), fi.z);
                        //atomicAdd(&(isspaf[atom2].x), fj.x);
                        //atomicAdd(&(isspaf[atom2].y), fj.y);
                        //atomicAdd(&(isspaf[atom2].z), fj.z);
                }
        }
}


/* C wrappers for kernels */
float isspa_force_cuda(float4 *xyz_d, float4 *f_d, float4 *isspaf_d, isspa& isspas, int nAtoms_h) {
        //float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {
        
        float milliseconds;
        
        // timing                                                                                                                
        cudaEventRecord(isspas.isspaStart);
        
	//cudaProfilerStart();
	// zero IS-SPA arrays on GPU
	cudaMemset(isspas.enow_d,  0.0f,  nAtoms_h*isspas.nMC*sizeof(float4));
	cudaMemset(isspas.e0now_d, 0.0f,  nAtoms_h*isspas.nMC*sizeof(float4));
	cudaMemset(isspaf_d,       0.0f,  nAtoms_h*sizeof(float4));
        
	// compute position of each MC point
	isspa_MC_points_kernel<<<isspas.mcGridSize,isspas.mcBlockSize >>>(xyz_d, isspas.mcpos_d, isspas.randStates_d, isspas.rmax_d, isspas.isspaTypes_d);
        // compute densities and mean electric field value for each MC point
	isspa_field_kernel<<<isspas.fieldGridSize, isspas.fieldBlockSize, nAtoms_h*sizeof(float4)>>>(xyz_d, isspas.rmax_d, isspas.vtot_d, isspas.isspaTypes_d, isspas.isspaGTable_d, isspas.isspaETable_d, isspas.enow_d, isspas.e0now_d, isspas.mcpos_d);
	// compute forces for each atom
	isspa_force_kernel<<<isspas.forceGridSize, isspas.forceBlockSize, nAtoms_h*sizeof(float4)>>>(xyz_d, isspas.rmax_d, isspas.isspaTypes_d, isspas.isspaForceTable_d, f_d, isspas.enow_d, isspas.e0now_d, isspas.mcpos_d, isspaf_d);
        
	//cudaDeviceSynchronize();
	//cudaProfilerStop();
	
        // finish timing
	cudaEventRecord(isspas.isspaStop);
	cudaEventSynchronize(isspas.isspaStop);
	cudaEventElapsedTime(&milliseconds, isspas.isspaStart, isspas.isspaStop);
	return milliseconds;
}

void isspa_grid_block(int nAtoms_h, int nPairs_h, float lbox_h, isspa& isspas) {
        
        float2 box_h;
	int maxThreadsPerBlock = 1024;

        // determine gridSize and blockSize for MC kernel	
	isspas.mcGridSize = int(ceil(isspas.nMC*nAtoms_h/(float) maxThreadsPerBlock));
	isspas.mcBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA mc kernel blocks: %d \n", isspas.mcGridSize);
	printf("Number of IS-SPA mc kernel threads per block: %d \n", isspas.mcBlockSize);

        
	// determine gridSize and blockSize for field kernel	
        //isspas.fieldThreads = 0;		
	isspas.fieldGridSize = nAtoms_h;
        isspas.fieldBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA field kernel blocks: %d \n", isspas.fieldGridSize);
	printf("Number of IS-SPA field kernel threads per block: %d \n", isspas.fieldBlockSize);
	
        // determine gridSize and blockSize for force kernel

        //isspas.forceThreads = int(ceil(nAtoms_h / 32.0))
	isspas.forceGridSize = int(nAtoms_h/32.0f)*nAtoms;
	isspas.forceBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA force kernel blocks: %d \n", isspas.forceGridSize);
	printf("Number of IS-SPA force kernel threads per block: %d \n", isspas.forceBlockSize);
	
	// fill box with box and half box length
	box_h.x = lbox_h;
	box_h.y = lbox_h/2.0f;
	
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
