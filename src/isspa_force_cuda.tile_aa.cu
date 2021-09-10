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
__global__ void __launch_bounds__(32, 8) isspa_field_kernel(float4 *xyz, const float* __restrict__ rmax, int *isspaTypes, const float* __restrict__  gTable, const float* __restrict__  eTable, float4 *enow, float4 *e0now, float4 *mcpos) { 
	int atom;
        int bin;
        int jt;
	int mc;
        int tRow, tCol;
        int tileIdx;
	float rmax_l;
	float dist2, dist;
        float fracDist;
        float g1, g2;
        float e1, e2;
        float etab;
        float2 gRparams_l = gRparams;
        float2 eRparams_l = eRparams;
        float4 r;
        float4 mcpos_l;
        float4 enow_l;
        float4 e0now_l;
       
        // Determine the tile index to be calculated
        tileIdx = blockIdx.x;
        //obtain the number of rows and columns in tile matrix 
        unsigned int nRows = ceil(double(nAtoms)/double(32.0));
        unsigned int nCols = ceil(double(nMC*nAtoms)/double(32.0f));
        // Determine the current tiles position in the tile matrix
        tRow = int(double(tileIdx)/double(nCols));
        tCol = tileIdx-nCols*tRow;
        // Determine the MC point for the thread
        mc = tCol*32 + threadIdx.x;
        // Load in atom data in shared memory                                                                                                                                                           
        extern __shared__ float4 xyz_s[];
        xyz_s[threadIdx.x] = xyz[32*tRow + threadIdx.x];
        __syncthreads();

        if (mc < nMC*nAtoms) {                                
                // zero the local variables that will be reduced
                mcpos_l = __ldg(mcpos+mc);
                mcpos_l.w = 1.0f;
                enow_l.x = enow_l.y = enow_l.z = enow_l.w = 0.0f;
                e0now_l.x = e0now_l.y = e0now_l.z = e0now_l.w = 0.0f;
                // loop over atoms in tile for each MC point in tile
                for (unsigned int offset = 0; offset < 32; offset++) {
                        atom =  32*tRow + offset;
                        // Load in position, atom type, and rmax of atom                                                                                                             
                        jt = __ldg(isspaTypes + atom);
                        rmax_l = rmax[jt];
                        if (atom < nAtoms) {
                                // Get constants for atom
                                jt = __ldg(isspaTypes+atom);
                                r = min_image(mcpos_l - xyz_s[offset],box.x,box.y);
                                dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
                                dist = sqrtf(dist2);
                                if (dist <= rmax_l) {
                                        e0now_l.w += 1.0f;
                                        // determine density bin of distance
                                        bin = int(__fdividef(dist-gRparams_l.x,gRparams_l.y)); 	
                                        // make sure bin is in limits of density table
                                        if (bin < 0) {
                                                mcpos_l.w = 0.0f;
                                        } else if (bin < nGRs-1) {
                                                // Push Density to MC point
                                                fracDist = __fdividef((dist - (gRparams_l.x+bin*gRparams_l.y)),gRparams_l.y);
                                                g1 = gTable[jt*nGRs+bin];
                                                g2 = gTable[jt*nGRs+bin+1];
                                                mcpos_l.w *= fmaf(g2,fracDist,g1*(1.0f-fracDist));
                                                //mcpos_l.w = gTable[jt*nGRs+bin];
                                                // Push mean field to MC point
                                                fracDist = __fdividef((dist - (eRparams_l.x+bin*eRparams_l.y)),eRparams_l.y);
                                                e1 = eTable[jt*nERs+bin];
                                                e2 = eTable[jt*nERs+bin+1];
                                                etab =  fmaf(e2,fracDist,e1*(1.0f-fracDist));
                                                //etab = eTable[jt*nGRs+bin];
                                                enow_l += r*__fdividef(etab,dist);
                                        }      
                                } else {
                                        e0now_l -= r*__fdividef(e0*xyz_s[offset].w,dist2*dist);
                                }				
                                enow_l -= r*__fdividef(e0*xyz_s[offset].w,dist2*dist);        
                        }                        
                }

                // Add the fields to the global variable
                atomicMul(&(mcpos[mc].w), mcpos_l.w);
                atomicAdd(&(enow[mc].x), enow_l.x);
                atomicAdd(&(enow[mc].y), enow_l.y);
                atomicAdd(&(enow[mc].z), enow_l.z);
                atomicAdd(&(e0now[mc].x), e0now_l.x);
                atomicAdd(&(e0now[mc].y), e0now_l.y);
                atomicAdd(&(e0now[mc].z), e0now_l.z);
                atomicAdd(&(e0now[mc].w), e0now_l.w);
        }
}

__global__ void __launch_bounds__(32, 2) isspa_force_kernel(float4 *xyz, const float* __restrict__ rmax, const float* __restrict__ vtot, int *isspaTypes, const float* __restrict__ forceTable, float4 *f, float4 *enow, float4 *e0now, float4 *mcpos, float4 *isspaf) {
        int bin;
        int jt;
        int mc;
        int atom;
        int tRow, tCol;
        int tileIdx;
        float igo;
        float fs;
        float r0;
        float rmax_l;
        float vtot_l;
        float dist2, dist;
        float pdotr;
        float cothE;
        float c1,c2,c3;
        float dp1,dp2,dp3;
        float Rz;
        float f1, f2;
        float fracDist;
        float4 xyz_l;
        float4 r;
        float4 fi;
        float4 fj;

        // Determine the tile index to be calculated
        tileIdx = blockIdx.x;

        //obtain the number of rows and columns in tile matrix
        unsigned int nRows = ceil(double(nAtoms)/double(32.0f));
        unsigned int nCols = ceil(double(nMC*nAtoms)/double(32.0f));

        // Determine the current tiles position in the tile matrix
        tRow = int(double(tileIdx)/double(nCols));
        tCol = tileIdx-nCols*tRow;

        // Determine the atom for the thread and lead in the parameters
        atom = 32*tRow + threadIdx.x;
        
        // Determine the atom parameters for this thread
        xyz_l = __ldg(xyz+atom);
        jt = __ldg(isspaTypes + atom);
        rmax_l = rmax[jt];
        vtot_l = vtot[jt];                                
        
        // Load in the density and electric fields for the MC points for this tile into shared memory
        __shared__ float4 mcpos_s[32];
        __shared__ float4 enow_s[32];
        __shared__ float4 e0now_s[32];        
        mc = tCol*32 + threadIdx.x;
        e0now_s[threadIdx.x] = e0now[mc];
        enow_s[threadIdx.x] = enow[mc];
        mcpos_s[threadIdx.x] = mcpos[mc];
        igo = __fdividef(vtot_l,e0now_s[threadIdx.x].w);
        mcpos_s[threadIdx.x].w *= igo;
        //if (isinf(mcpos_s[threadIdx.x].w)) {
        //        printf("atom: %d mc: %d dens: %f %f igo: %f e0now: %f\n",atom,mc,mcpos_s[threadIdx.x].w,mcpos[mc].w,igo,e0now_s[threadIdx.x].w);
        //}
        r0 = norm3df(enow_s[threadIdx.x].x, enow_s[threadIdx.x].y, enow_s[threadIdx.x].z);
        enow_s[threadIdx.x].x = __fdividef(enow_s[threadIdx.x].x,r0);
        enow_s[threadIdx.x].y = __fdividef(enow_s[threadIdx.x].y,r0);
        enow_s[threadIdx.x].z = __fdividef(enow_s[threadIdx.x].z,r0);
        enow_s[threadIdx.x].w = r0;
        e0now_s[threadIdx.x].x = __fdividef(e0now_s[threadIdx.x].x,3.0f);
        e0now_s[threadIdx.x].y = __fdividef(e0now_s[threadIdx.x].y,3.0f);
        e0now_s[threadIdx.x].z = __fdividef(e0now_s[threadIdx.x].z,3.0f);
        e0now_s[threadIdx.x].w = igo;        

        __syncthreads();


        if (atom < nAtoms) {
                
                // Zero out the forces
                fi.x = fi.y = fi.z = 0.0f;
                fj.x = fj.y = fj.z = 0.0f;
        
                // loop over the MC points for each atom 
                for (unsigned int offset = 0; offset < 32; offset += 1) {
                        // Determine the MC points from atom2
                        mc = tCol*32 + offset;
                        if (mc < nMC*nAtoms) {
                                // Calculate the distance between the MC point and atom1
                                r = min_image(mcpos_s[offset] - xyz_l,box.x,box.y);
                                dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
                                dist = sqrtf(dist2);                
                                // Coulombic Force
                                cothE=__fdividef(1.0f,tanhf(enow_s[offset].w));
                                c1=cothE-__fdividef(1.0f,enow_s[offset].w);
                                c2=1.0f-2.0f*__fdividef(c1,enow_s[offset].w);
                                c3=cothE-3.0f*__fdividef(c2,enow_s[offset].w);                
                                Rz=__fdividef(enow_s[offset].x*r.x+enow_s[offset].y*r.y+enow_s[offset].z*r.z,dist);
                                dp1=3.0f*Rz;
                                dp2=7.5f*Rz*Rz-1.5f;
                                dp3=(17.50f*Rz*Rz-7.50f)*Rz;                
                                // Calculate dipole term
                                fs = __fdividef(-xyz_l.w*p0*c1*mcpos_s[offset].w,dist2*dist);
                                fi += fs*(r*__fdividef(dp1,dist)-enow_s[offset]);
                                //fj += fs*(r*__fdividef(dp1,dist)-enow_s[offset]);
                                // Calculate quadrapole term
                                fs = __fdividef(-xyz_l.w*q0*(1.5f*c2-0.5f)*mcpos_s[offset].w,dist2*dist2);
                                fi += fs*(r*__fdividef(dp2,dist)-dp1*enow_s[offset]);
                                //fj += fs*(r*__fdividef(dp2,dist)-dp1*enow_s[offset]);
                                // Calculate octapole term
                                fs = __fdividef(-xyz_l.w*o0*(2.5f*c3-1.5f*c1)*mcpos_s[offset].w,dist2*dist2*dist);
                                fi += fs*(r*__fdividef(dp3,dist)-dp2*enow_s[offset]);
                                //fj += fs*(r*__fdividef(dp3,dist)-dp2*enow_s[offset]);
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
                                                fs = (f1*(1.0-fracDist)+f2*fracDist)*mcpos_s[offset].w;
                                                fs =  fmaf(f2,fracDist,f1*(1.0f-fracDist))*mcpos_s[offset].w;
                                                //fs = forceTable[jt*nRs+bin]*mcpos_s[offset].w;
                                        }
                                        fi += r*__fdividef(-fs,dist);
                                        //fj += r*__fdividef(-fs,dist);

                                } else {
                                        // Constant Density Dielectric
                                        fs=__fdividef(-xyz_l.w*p0,dist2*dist);
                                        pdotr=__fdividef(3.0f*(e0now_s[offset].x*r.x+e0now_s[offset].y*r.y+e0now_s[offset].z*r.z),dist2);
                                        fi -= fs*(pdotr*r-e0now_s[offset])*e0now_s[offset].w;
                                        fj -= fs*(pdotr*r-e0now_s[offset])*e0now_s[offset].w;
                                }	
                        }
                }
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
	// zero IS-SPA arrays on GPU
	cudaMemset(isspas.enow_d,  0.0f,  nAtoms_h*isspas.nMC*sizeof(float4));
	cudaMemset(isspas.e0now_d, 0.0f,  nAtoms_h*isspas.nMC*sizeof(float4));
	cudaMemset(isspaf_d,       0.0f,  nAtoms_h*sizeof(float4));
        
	// compute position of each MC point
	isspa_MC_points_kernel<<<isspas.mcGridSize,isspas.mcBlockSize >>>(xyz_d, isspas.mcpos_d, isspas.randStates_d, isspas.rmax_d, isspas.isspaTypes_d);
        // compute densities and mean electric field value for each MC point
	isspa_field_kernel<<<isspas.fieldGridSize, isspas.fieldBlockSize, 32*sizeof(float4)>>>(xyz_d, isspas.rmax_d, isspas.isspaTypes_d, isspas.isspaGTable_d, isspas.isspaETable_d, isspas.enow_d, isspas.e0now_d, isspas.mcpos_d);
	// compute forces for each atom
        isspa_force_kernel<<<isspas.forceGridSize, isspas.forceBlockSize>>>(xyz_d,isspas.rmax_d,isspas.vtot_d,isspas.isspaTypes_d,isspas.isspaForceTable_d,f_d,isspas.enow_d,isspas.e0now_d,isspas.mcpos_d,isspaf_d);
        
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

        // determine gridSize and blockSize for MC kernel	
	isspas.mcGridSize = int(ceil(isspas.nMC*nAtoms_h/ (float) maxThreadsPerBlock));
	isspas.mcBlockSize = maxThreadsPerBlock;
	printf("Number of IS-SPA mc kernel blocks: %d \n", isspas.mcGridSize);
	printf("Number of IS-SPA mc kernel threads per block: %d \n", isspas.mcBlockSize);

        
	// determine gridSize and blockSize for field kernel	
	isspas.fieldGridSize = ceil(nAtoms_h/32.0)*ceil(isspas.nMC*nAtoms_h/32.0);
        isspas.fieldBlockSize = 32;
	printf("Number of IS-SPA field kernel blocks: %d \n", isspas.fieldGridSize);
	printf("Number of IS-SPA field kernel threads per block: %d \n", isspas.fieldBlockSize);
	
        // determine gridSize and blockSize for force kernel

        //isspas.forceThreads = temp*32;		
	isspas.forceGridSize = ceil(nAtoms_h/32.0)*ceil(isspas.nMC*nAtoms_h/32.0);
	isspas.forceBlockSize = 32;
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
