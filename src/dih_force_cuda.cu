
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "dih_class.h"
#include "dih_force_cuda.h"
#include "constants.h"

// CUDA Kernels

__global__ void dih_force_kernel(float4 *xyz, float4 *f, int nAtoms, float lbox, int4 *dihAtoms, int *dihTypes, float4 *dihParams, int nDihs, int nTypes, float2 *scaled14Factors, float2 *lj, int *atomType, int *nbparm, int nAtomTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	//unsigned int t = threadIdx.x;
	//extern __shared__ float4 dihParams_s[];
	//extern __shared__ float2 scaled14_s[];
//	extern __shared__ float scnb_s[];
	int4 atoms;
	int dihType;
	float4 p1, p2, p3, p4;
	float4 r14;
	float4 r12;
	float4 r23;
	float4 r34;
	float c11, c22, c33, c12, c13, c23;
	float t1, t2, t3, t4, t5, t6;
	float a, b;
	float4 f1, f4;
	float phi;
	float fdih;
	float hbox;
	float rMag, r6;
	//int it, jt,
	int nlj;
	float f14e,f14v;
	float4 params;
	float2 ljAB;
	float2 sc14;
	//int i;

	// copy dihedral parameters to shared memory per block
//	for (i=t;i<nTypes;i+=blockDim.x) {
//		dihParams_s[i] = __ldg(dihParams+i);
//		scaled14_s[i] = __ldg(scaled14Factors+i);
//	}
//	__syncthreads();
	if (index < nDihs)
	{
		hbox = 0.5f*lbox;
		// determine four atoms involved in dihderal
		atoms = __ldg(dihAtoms+index);
		p1 = __ldg(xyz+atoms.x);   // position of atom 1
		p2 = __ldg(xyz+atoms.y);   // position of atom 2
		p3 = __ldg(xyz+abs(atoms.z));   // position of atom 3
		p4 = __ldg(xyz+abs(atoms.w));   // position of atom 4
		// determine dihedral type
		dihType = __ldg(dihTypes+index);
		// Check to see if we want to compute the scaled 1-4 interaction
		if (atoms.z > 0 && atoms.w > 0) {
			//Scaled non-bonded interaction for 1-4
			// load atom type data
			nlj = __ldg(nbparm + nAtomTypes*__ldg(atomType+atoms.x) + __ldg(atomType+atoms.w));
			sc14 = __ldg(scaled14Factors + dihType);
			ljAB = __ldg(lj+nlj);
			// compute r between 1 and 4
			r14 = min_image(p1-p4,lbox,hbox);
			rMag = r14.x*r14.x+r14.y*r14.y+r14.z*r14.z;
			// compute scaled non-bond force
			r6 = powf(rMag,-3.0f);
			//f14e = p1.w*p4.w/rMag/sqrtf(rMag)/sc14.x;
			f14e = __fdividef(p1.w*p4.w,rMag*sqrtf(rMag)*sc14.x);
			//f14v = r6*(12.0f*ljAB.x*r6-6.0f*ljAB.y)/sc14.y/rMag;
			f14v = __fdividef(r6*(12.0f*ljAB.x*r6-6.0f*ljAB.y),sc14.y*rMag);
			// add scaled non-bond force to atoms 1 and 4
			atomicAdd(&(f[atoms.x].x), (f14e+f14v)*r14.x);
			atomicAdd(&(f[atoms.w].x), -(f14e+f14v)*r14.x);
			atomicAdd(&(f[atoms.x].y), (f14e+f14v)*r14.y);
			atomicAdd(&(f[atoms.w].y), -(f14e+f14v)*r14.y);
			atomicAdd(&(f[atoms.x].z), (f14e+f14v)*r14.z);
			atomicAdd(&(f[atoms.w].z), -(f14e+f14v)*r14.z);
		}
		if (atoms.w < 0) { atoms.w = -atoms.w;} // atom4 is negative if the torsion is improper
		if (atoms.z < 0) { atoms.z = -atoms.z;} // atom3 is negative if we don't want to compute the scaled 1-4 for this torsion

		r12 = min_image(p1-p2,lbox,hbox);
		r23 = min_image(p2-p3,lbox,hbox);
		r34 = min_image(p3-p4,lbox,hbox);
		// dot products
		c11 = r12.x*r12.x + r12.y*r12.y + r12.z*r12.z;
		c22 = r23.x*r23.x + r23.y*r23.y + r23.z*r23.z;
		c12 = r12.x*r23.x + r12.y*r23.y + r12.z*r23.z;
		c33 = r34.x*r34.x + r34.y*r34.y + r34.z*r34.z;
		c23 = r23.x*r34.x + r23.y*r34.y + r23.z*r34.z;
		c13 = r12.x*r34.x + r12.y*r34.y + r12.z*r34.z;
		// cross
		t1 = c13*c22-c12*c23;
		t2 = c11*c23-c12*c13;
		t3 = c12*c12-c11*c22;
		t4 = c22*c33-c23*c23;
		t5 = c13*c23-c12*c33;
		t6 = -t1;

		b = sqrtf(-t3*t4);
		a = t6/b;
		// make sure a is in the domain of the arccos
		if (a <= -1.0f) {
			fdih = 0.0f;
		} else if (a >= 1.0f) {
			fdih = 0.0f;	
		} else {
			phi = acos(a);
			params = __ldg(dihParams+dihType);
			//params = dihParams_s[dihType];
			//fdih = params.x * params.y * sinf(params.x*phi-params.z)/sinf(phi)*c22/b;
			fdih = __fdividef(params.x * params.y * sinf(params.x*phi-params.z)*c22,sinf(phi)*b);
		}
		// force components
		f1= __fdividef(fdih,t3)*(t1*r12+t2*r23+t3*r34);
		f4= __fdividef(-fdih,t4)*(t4*r12+t5*r23+t6*r34);
		// add forces to atoms
		atomicAdd(&(f[atoms.x].x), f1.x);
		atomicAdd(&(f[atoms.y].x), -(1.0f+__fdividef(c12,c22))*f1.x+__fdividef(c23,c22)*f4.x);
		atomicAdd(&(f[atoms.z].x), __fdividef(c12,c22)*f1.x-(1.0f+__fdividef(c23,c22))*f4.x);
		atomicAdd(&(f[atoms.w].x), f4.x);
		atomicAdd(&(f[atoms.x].y), f1.y);
		atomicAdd(&(f[atoms.y].y), -(1.0f+__fdividef(c12,c22))*f1.y+__fdividef(c23,c22)*f4.y);
		atomicAdd(&(f[atoms.z].y), __fdividef(c12,c22)*f1.y-(1.0f+__fdividef(c23,c22))*f4.y);
		atomicAdd(&(f[atoms.w].y), f4.y);
		atomicAdd(&(f[atoms.x].z), f1.z);
		atomicAdd(&(f[atoms.y].z), -(1.0f+__fdividef(c12,c22))*f1.z+__fdividef(c23,c22)*f4.z);
		atomicAdd(&(f[atoms.z].z), __fdividef(c12,c22)*f1.z-(1.0f+__fdividef(c23,c22))*f4.z);
		atomicAdd(&(f[atoms.w].z), f4.z);

	}
}

/* C wrappers for kernels */

float dih_force_cuda(atom& atoms, dih& dihs, float lbox)
{
	float milliseconds;
	
	// timing
	cudaEventRecord(dihs.dihStart);

	// run dih cuda kernel
	//dih_force_kernel<<<dihs.gridSize, dihs.blockSize, dihs.nTypes*sizeof(float4) >>>(atoms.pos_d, atoms.for_d, atoms.nAtoms, lbox, dihs.dihAtoms_d, dihs.dihTypes_d, dihs.dihParams_d, dihs.nDihs, dihs.nTypes, dihs.scaled14Factors_d, atoms.lj_d, atoms.ityp_d, atoms.nonBondedParmIndex_d, atoms.nTypes); 
	dih_force_kernel<<<dihs.gridSize, dihs.blockSize >>>(atoms.pos_d, atoms.for_d, atoms.nAtoms, lbox, dihs.dihAtoms_d, dihs.dihTypes_d, dihs.dihParams_d, dihs.nDihs, dihs.nTypes, dihs.scaled14Factors_d, atoms.lj_d, atoms.ityp_d, atoms.nonBondedParmIndex_d, atoms.nTypes); 

	// finalize timing
	cudaEventRecord(dihs.dihStop);
	cudaEventSynchronize(dihs.dihStop);
	cudaEventElapsedTime(&milliseconds, dihs.dihStart, dihs.dihStop);
	return milliseconds;


}
extern "C" void dih_force_cuda_grid_block(int nDihs, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, dih_force_kernel, 0, nDihs); 

    	// Round up according to array size 
    	*gridSize = (nDihs + *blockSize - 1) / *blockSize; 
}
