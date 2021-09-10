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
__global__ void isspa_field_kernel(float4 *xyz, const float* __restrict__ rmax, int *isspaTypes, const float* __restrict__  gTable, const float* __restrict__  eTable, float4 *enow, float4 *e0now, float4 *mcpos, int nThreads) { 
        unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom;
        int atom2;
	int MC;
	int MCind;
	int bin;
        int it;
	int jt;
        float rmax_l;
        float dist2, dist;
        float fracDist;
        float g1, g2;
        float e1, e2;
        float etab;
        float2 gRparams_l = gRparams;
        float2 eRparams_l = eRparams;
	float4 atom2_pos;
        float4 r;
        float4 mcpos_l;
        float4 enow_l;
        float4 e0now_l;
	// Determine which atom the MC point is being generated on
	atom = int(double(index)/double(nThreads*nMC));
        MC = int(double(index)/double(nThreads));
	MCind = int(MC - atom*nMC);
	atom2 = int(index - atom*nMC*nThreads - MCind*nThreads);
	// zero the local variables that will be reduced
	if (atom < nAtoms) {
	        if (MCind < nMC) {
		        if (atom2 < nAtoms) {
                                enow_l.x = enow_l.y = enow_l.z = enow_l.w = 0.0f;
                                e0now_l.x = e0now_l.y = e0now_l.z = e0now_l.w = 0.0f;
                                // Get atom positions
                                mcpos_l = __ldg(mcpos+MC);
                                it = __ldg(isspaTypes+atom);
                                rmax_l = rmax[it];                                
				// Get atom positions
				atom2_pos = __ldg(xyz+atom2);
				// Get constants for atom
				jt = __ldg(isspaTypes+atom2);
				r = min_image(mcpos_l - atom2_pos,box.x,box.y);
				dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
				dist = sqrtf(dist2);
                                if (dist <= rmax_l) {
				        e0now_l.w = 1.0f;
                                        // determine density bin of distance
                                        bin = int(__fdividef(dist-gRparams_l.x,gRparams_l.y)); 	
                                        // make sure bin is in limits of density table
                                        if (bin < 0) {
					        mcpos_l.w = 0.0f;
					} else if (bin < nGRs-1) {
					        // Push Density to MC point
					        //fracDist = (dist - (gRparams_l.x+bin*gRparams_l.y)) / gRparams_l.y;
                                                fracDist = __fdividef((dist - (gRparams_l.x+bin*gRparams_l.y)),gRparams_l.y);
                                                g1 = gTable[jt*nGRs+bin];
                                                g2 = gTable[jt*nGRs+bin+1];
                                                mcpos_l.w = fmaf(g2,fracDist,g1*(1.0f-fracDist));
                                                //mcpos_l.w = gTable[jt*nGRs+bin];
                                                // Push mean field to MC point
						//fracDist = (dist - (eRparams_l.x+bin*eRparams_l.y)) / eRparams_l.y;
						fracDist = __fdividef((dist - (eRparams_l.x+bin*eRparams_l.y)),eRparams_l.y);
                                                e1 = eTable[jt*nERs+bin];
                                                e2 = eTable[jt*nERs+bin+1];
                                                etab =  fmaf(e2,fracDist,e1*(1.0f-fracDist));
                                                //etab = eTable[jt*nGRs+bin];
                                                enow_l += r*__fdividef(etab,dist);
                                        }      
                                } else {
                                        e0now_l = -r*__fdividef(e0*atom2_pos.w,dist2*dist);
                                        e0now_l.w = 0.0f;
                                        mcpos_l.w = 1.0f;						
				}				
				enow_l -= r*__fdividef(e0*atom2_pos.w,dist2*dist);
                        } else {
                                enow_l.x = enow_l.y = enow_l.z = 0.0f;
                                e0now_l.x = e0now_l.y = e0now_l.z = e0now_l.w = 0.0f;
                                mcpos_l.w = 1.0f;
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
}

__global__ void isspa_force_kernel(float4 *xyz, const float* __restrict__ vtot, const float* __restrict__ rmax, int *isspaTypes, const float* __restrict__ forceTable, float4 *f, float4 *enow, float4 *e0now, float4 *mcpos, float nThreads, float4 *isspaf) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int bin;
        int jt;
	int MC;
	int atom;
	float igo;
	float fs;
	float r0;
        float rmax_l;
	float vtot_l;
        float dist2, dist;
        float pdotr;
	float cothE;
	float c1,c2,c3;
	float  dp1,dp2,dp3;
	float Rz;
        float f1, f2;
        float fracDist;
	float4 xyz_l;
        float4 r;
        float4 fi;
        //float4 fj;
	float4 mcpos_l;
	float4 enow_l;
	float4 e0now_l;

	// Determine the atom for which the force is being summed on
	atom = int(double(index)/double(nThreads));        
	MC = int(index-atom*nThreads);
        // Zero out the forces
	fi.x = fi.y = fi.z = 0.0f;
	//fj.x = fj.y = fj.z = 0.0f;
        if (atom < nAtoms) {
                if (MC < nAtoms*nMC) {       
                        // Load in position, atom type, and rmax of atom
                        xyz_l = __ldg(xyz+atom);
                        jt = __ldg(isspaTypes + atom);
                        rmax_l = rmax[jt];
                        vtot_l = vtot[jt];
                        // Load in field data for the MC point	
                        mcpos_l = __ldg(mcpos+MC);
                        enow_l = __ldg(enow+MC);
                        e0now_l = __ldg(e0now+MC);
                        // Finish calculating density
                        igo = __fdividef(vtot_l,e0now_l.w);
                        mcpos_l.w *= igo;
                        // Convert enow into polarzation
                        r0 = norm3df(enow_l.x, enow_l.y, enow_l.z);
                        enow_l.x = __fdividef(enow_l.x,r0);			
                        enow_l.y = __fdividef(enow_l.y,r0);			
                        enow_l.z = __fdividef(enow_l.z,r0);			
                        enow_l.w = r0;
                        e0now_l.x = __fdividef(e0now_l.x,3.0f);
                        e0now_l.y = __fdividef(e0now_l.y,3.0f);
                        e0now_l.z = __fdividef(e0now_l.z,3.0f);
                        e0now_l.w = igo;               
                        // Calculate the distance between the MC point and atom1
                        r = min_image(mcpos_l - xyz_l,box.x,box.y);
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
                        fs = __fdividef(-xyz_l.w*p0*c1*mcpos_l.w,dist2*dist);
                        fi += fs*(r*__fdividef(dp1,dist)-enow_l);
                        //fj += fs*(r*__fdividef(dp1,dist)-enow_l);
                        // Calculate quadrapole term
                        fs = __fdividef(-xyz_l.w*q0*(1.5f*c2-0.5f)*mcpos_l.w,dist2*dist2);
                        fi += fs*(r*__fdividef(dp2,dist)-dp1*enow_l);
                        //fj += fs*(r*__fdividef(dp2,dist)-dp1*enow_l);
                        // Calculate octapole term
                        fs = __fdividef(-xyz_l.w*o0*(2.5f*c3-1.5f*c1)*mcpos_l.w,dist2*dist2*dist);
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
                                        f1 = forceTable[jt*nRs+bin];
                                        f2 = forceTable[jt*nRs+bin+1];
                                        fs = (f1*(1.0-fracDist)+f2*fracDist)*mcpos_l.w;
                                        fs =  fmaf(f2,fracDist,f1*(1.0f-fracDist))*mcpos_l.w;
                                        //fs = forceTable[jt*nRs+bin]*mcpos_l.w;
                                }
                                fi += r*__fdividef(-fs,dist);
                                //fj += r*__fdividef(-fs,dist);
                        } else {
                                // Constant Density Dielectric
                                fs=__fdividef(-xyz_l.w*p0,dist2*dist);
                                pdotr=__fdividef(3.0f*(e0now_l.x*r.x+e0now_l.y*r.y+e0now_l.z*r.z),dist2);
                                fi -= fs*(pdotr*r-e0now_l)*e0now_l.w;
                                //fj -= fs*(pdotr*r-e0now_l)*e0now_l.w;
                        }	
                }
        }
        // Warp reduce the forces
	fi =  warpReduceSumTriple(fi);
        //fj =  warpReduceSumTriple(fj);
        // Add the force to the global force
	if ((threadIdx.x & (warpSize - 1)) == 0) {
                atomicAdd(&(f[atom].x), fi.x);
                atomicAdd(&(f[atom].y), fi.y);
                atomicAdd(&(f[atom].z), fi.z);
                //atomicAdd(&(isspaf[atom].x), fj.x);
                //atomicAdd(&(isspaf[atom].y), fj.y);
                //atomicAdd(&(isspaf[atom].z), fj.z);
	}
}


/* C wrappers for kernels */
float isspa_force_cuda(float4 *xyz_d, float4 *f_d, float4 *isspaf_d, isspa& isspas, int nAtoms_h) {
        //float isspa_force_cuda(float4 *xyz_d, float4 *f_d, isspa& isspas, int nAtoms_h) {
        
        float milliseconds;
        
        // timing                                                                                                                
        cudaEventRecord(isspas.isspaStart);
        
	cudaProfilerStart();
	// zero IS-SPA arrays on GPU
	cudaMemset(isspas.enow_d,  0.0f,  nAtoms_h*isspas.nMC*sizeof(float4));
	cudaMemset(isspas.e0now_d, 0.0f,  nAtoms_h*isspas.nMC*sizeof(float4));
	//cudaMemset(isspaf_d,       0.0f,  nAtoms_h*sizeof(float4));
        
	// compute position of each MC point
	isspa_MC_points_kernel<<<isspas.mcGridSize,isspas.mcBlockSize >>>(xyz_d, isspas.mcpos_d, isspas.randStates_d, isspas.rmax_d, isspas.isspaTypes_d);
        // compute densities and mean electric field value for each MC point
	isspa_field_kernel<<<isspas.fieldGridSize, isspas.fieldBlockSize>>>(xyz_d, isspas.rmax_d, isspas.isspaTypes_d, isspas.isspaGTable_d, isspas.isspaETable_d, isspas.enow_d, isspas.e0now_d, isspas.mcpos_d, isspas.fieldThreads);
	// compute forces for each atom
	isspa_force_kernel<<<isspas.forceGridSize, isspas.forceBlockSize>>>(xyz_d,isspas.vtot_d,isspas.rmax_d,isspas.isspaTypes_d,isspas.isspaForceTable_d,f_d,isspas.enow_d,isspas.e0now_d,isspas.mcpos_d,isspas.forceThreads,isspaf_d);
        
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

        // determine gridSize and blockSize for MC kernel	
	isspas.mcGridSize = int(ceil(isspas.nMC*nAtoms_h/(float) maxThreadsPerBlock));
	isspas.mcBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA mc kernel blocks: %d \n", isspas.mcGridSize);
	printf("Number of IS-SPA mc kernel threads per block: %d \n", isspas.mcBlockSize);

        
	// determine gridSize and blockSize for field kernel	
	temp = int(ceil((nAtoms_h) / (float) 32.0));
	isspas.fieldThreads = temp*32;		
	isspas.fieldGridSize = int(ceil(isspas.fieldThreads*nAtoms_h*isspas.nMC / (float) maxThreadsPerBlock));
	isspas.fieldBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA field kernel blocks: %d \n", isspas.fieldGridSize);
	printf("Number of IS-SPA field kernel threads per block: %d \n", isspas.fieldBlockSize);
	
        // determine gridSize and blockSize for force kernel
	temp = int(ceil((nAtoms_h*isspas.nMC) / (float) 32.0));
	isspas.forceThreads = temp*32;		
	isspas.forceGridSize = int(ceil(isspas.forceThreads*nAtoms_h / (float) maxThreadsPerBlock));
	isspas.forceBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA force kernel blocks: %d \n", isspas.forceGridSize);
	printf("Number of IS-SPA force kernel threads per block: %d \n", isspas.forceBlockSize);
	printf("Number of IS-SPA force kernel MC points: %d %d \n", isspas.forceThreads,nAtoms_h*isspas.nMC);
	
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
