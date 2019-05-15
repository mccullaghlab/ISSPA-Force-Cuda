
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "atom_class.h"
#include "dih_class.h"
#include "dih_force_cuda.h"
#include "constants.h"

// CUDA Kernels

//__global__ void dih_force_kernel(float *xyz, float *f, int nAtoms, float lbox, int *dihAtoms, float *dihKs, float *dihNs, float *dihPs, int nDihs, float *scee, float *scnb, float *charge, float *ljA, float *ljB, int *atomType, int *nbparm, int nAtomTypes) {
__global__ void dih_force_kernel(float *xyz, float *f, int nAtoms, float lbox, int4 *dihAtoms, int *dihTypes, float4 *dihParams, int nDihs, int nTypes, float *scee, float *scnb, float *charge, float2 *lj, int *atomType, int *nbparm, int nAtomTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	//unsigned int t = threadIdx.x;
	//extern __shared__ float4 dihParams_s[];
	int4 atoms;
	int dihType;
	int  k;
	float r1[nDim];
	float r2[nDim];
	float r3[nDim];
	float c11, c22, c33, c12, c13, c23;
	float t1, t2, t3, t4, t5, t6;
	float a, b;
	float f1, f4;
	float phi;
	float fdih;
	float hbox;
	float rMag, r6;
	int it, jt,  nlj;
	float f14e,f14v;
	float4 params;
	float2 ljAB;
	//int chunk;

	// copy dihedral parameters to shared memory per block
	//chunk = (int) ( (nTypes + blockDim.x - 1)/blockDim.x);	
	//if ((t+1)*chunk <= nTypes)
	//{
	//	for (i=t*chunk;i<(t+1)*chunk;i++) {
	//		dihParams_s[i] = __ldg(dihParams+i);
	//	}
	//}
	//__syncthreads();
	if (index < nDihs)
	{
		hbox = lbox/2.0;
		// determine four atoms involved in dihderal
		//atom1 = __ldg(dihAtoms+index*5);
		//atom2 = __ldg(dihAtoms+index*5+1);
		//atom3 = __ldg(dihAtoms+index*5+2);
		//atom4 = __ldg(dihAtoms+index*5+3);
		atoms = __ldg(dihAtoms+index);
		// determine dihedral type
		dihType = __ldg(dihTypes+index);
		// Check to see if we want to compute the scaled 1-4 interaction
		if (atoms.z > 0 && atoms.w > 0) {
			//Scaled non-bonded interaction for 1-4
			rMag = 0.0f;
			for (k=0;k<nDim;k++) {
				r1[k] = __ldg(xyz+atoms.x+k)-__ldg(xyz+atoms.z+k);
				rMag += r1[k] * r1[k];
			}
			r6 = rMag*rMag*rMag;
			r6 = 1.0/r6;
			it = __ldg(atomType+atoms.x/3);
			jt = __ldg(atomType+atoms.w/3);
			nlj = nAtomTypes * (it-1) + jt - 1;
			nlj = __ldg(nbparm+nlj);
			f14e = __ldg(charge+atoms.x/3)*__ldg(charge+atoms.w/3)/rMag/sqrtf(rMag)/__ldg(scee+dihType);
			ljAB = __ldg(lj+nlj);
			f14v = r6*(12.0f*ljAB.x*r6-6.0f*ljAB.y)/__ldg(scnb+dihType)/rMag;
			f14v = 0.0f;
			for (k=0;k<nDim;k++) {
				atomicAdd(&f[atoms.x+k], (f14e+f14v)*r1[k]);
				atomicAdd(&f[atoms.w+k], -(f14e+f14v)*r1[k]);
			}
		}
		if (atoms.w < 0) { atoms.w = -atoms.w;} // atom4 is negative if the torsion is improper
		if (atoms.z < 0) { atoms.z = -atoms.z;} // atom3 is negative if we don't want to compute the scaled 1-4 for this torsion

		c11 = 0.0f;
		c22 = 0.0f;
		c33 = 0.0f;
		c12 = 0.0f;
		c13 = 0.0f;
		c23 = 0.0f;
		for (k=0;k<nDim;k++) {
			r1[k] = __ldg(xyz+atoms.x+k) - __ldg(xyz+atoms.y+k);
			r2[k] = __ldg(xyz+atoms.y+k) - __ldg(xyz+atoms.z+k);
			r3[k] = __ldg(xyz+atoms.z+k) - __ldg(xyz+atoms.w+k);
			// assuming no more than one box away
			if (r1[k] > hbox) {
				r1[k] -= lbox;
			} else if (r1[k] < -hbox) {
				r1[k] += lbox;
			}
			if (r2[k] > hbox) {
				r2[k] -= lbox;
			} else if (r2[k] < -hbox) {
				r2[k] += lbox;
			}	
			if (r3[k] > hbox) {
				r3[k] -= lbox;
			} else if (r3[k] < -hbox) {
				r3[k] += lbox;
			}	
			c11 += r1[k]*r1[k];
			c22 += r2[k]*r2[k];
			c12 += r1[k]*r2[k];
			c33 += r3[k]*r3[k];
			c23 += r2[k]*r3[k];
			c13 += r1[k]*r3[k];
		}
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
			fdih = 0.0;
		} else if (a >= 1.0f) {
			fdih = 0.0;	
		} else {
			phi = acos(a);
			params = __ldg(dihParams+dihType);
			//fdih = dihParams_s[dihType].x * dihParams_s[dihType].y * sinf(dihParams_s[dihType].x*phi-dihParams_s[dihType].z)/sinf(phi)*c22/b;
			fdih = params.x * params.y * sinf(params.x*phi-params.z)/sinf(phi)*c22/b;
			//fdih = __ldg(dihNs+dihType) * __ldg(dihKs+dihType) * sinf(__ldg(dihNs+dihType)*phi-__ldg(dihPs+dihType))/sinf(phi)*c22/b;
		}
		for (k=0;k<3;k++) {
			f1=fdih*(t1*r1[k]+t2*r2[k]+t3*r3[k])/t3;
			f4=-fdih*(t4*r1[k]+t5*r2[k]+t6*r3[k])/t4;
			atomicAdd(&f[atoms.x+k], f1);
			atomicAdd(&f[atoms.y+k], -(1.0f+c12/c22)*f1+c23/c22*f4);
			atomicAdd(&f[atoms.z+k], c12/c22*f1-(1.0f+c23/c22)*f4);
			atomicAdd(&f[atoms.w+k], f4);
		}

	}
}

/* C wrappers for kernels */

float dih_force_cuda(atom& atoms, dih& dihs, float lbox)
{
	float milliseconds;
	
	// timing
	cudaEventRecord(dihs.dihStart);

	// run dih cuda kernel
	//dih_force_kernel<<<dihs.gridSize, dihs.blockSize>>>(atoms.xyz_d, atoms.f_d, atoms.nAtoms, lbox, dihs.dihAtoms_d, dihs.dihKs_d, dihs.dihNs_d, dihs.dihPs_d, dihs.nDihs, dihs.sceeScaleFactor_d, dihs.scnbScaleFactor_d, atoms.charges_d, atoms.ljA_d, atoms.ljB_d, atoms.ityp_d, atoms.nonBondedParmIndex_d, atoms.nTypes);
	//dih_force_kernel<<<dihs.gridSize, dihs.blockSize, dihs.nTypes*sizeof(float4)>>>(atoms.xyz_d, atoms.f_d, atoms.nAtoms, lbox, dihs.dihAtoms_d, dihs.dihParams_d, dihs.nDihs, dihs.nTypes, dihs.sceeScaleFactor_d, dihs.scnbScaleFactor_d, atoms.charges_d, atoms.ljA_d, atoms.ljB_d, atoms.ityp_d, atoms.nonBondedParmIndex_d, atoms.nTypes);
	dih_force_kernel<<<dihs.gridSize, dihs.blockSize>>>(atoms.xyz_d, atoms.f_d, atoms.nAtoms, lbox, dihs.dihAtoms_d, dihs.dihTypes_d, dihs.dihParams_d, dihs.nDihs, dihs.nTypes, dihs.sceeScaleFactor_d, dihs.scnbScaleFactor_d, atoms.charges_d, atoms.lj_d, atoms.ityp_d, atoms.nonBondedParmIndex_d, atoms.nTypes);

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
