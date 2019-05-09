
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "dih_force_cuda.h"
#include "constants.h"

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// CUDA Kernels

__global__ void dih_force_kernel(float *xyz, float *f, int nAtoms, float lbox, int *dihAtoms, float *dihKs, float *dihNs, float *dihPs, int nDihs, float *scee, float *scnb, float *charge, float *ljA, float *ljB, int *atomType, int *nbparm, int nAtomTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
//	extern __shared__ float xyz_s[];
	int atom1;
	int atom2;
	int atom3;
	int atom4;
	int dihType;
	int k;
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
	
	//if (t < nAtoms*nDim) {
	//	xyz_s[t] = xyz[t];	
	//	__syncthreads();
	//}

	if (index < nDihs)
	{
		hbox = lbox/2.0;
		// determine two atoms to work  - these will be unique to each index
		atom1 = dihAtoms[index*5];
		atom2 = dihAtoms[index*5+1];
		atom3 = dihAtoms[index*5+2];
		atom4 = dihAtoms[index*5+3];
		dihType = dihAtoms[index*5+4];
		// Check to see if we want to compute the scaled 1-4 interaction
		if (atom3 > 0 && atom4 > 0) {
			//Scaled non-bonded interaction for 1-4
			rMag = 0.0f;
			for (k=0;k<nDim;k++) {
				r1[k] = xyz[atom1+k]-xyz[atom4+k];
				rMag += r1[k] * r1[k];
			}
			r6 = rMag*rMag*rMag;
			r6 = 1.0/r6;
			it = atomType[atom1];
			jt = atomType[atom4];
			nlj = nAtomTypes * (it-1) + jt - 1;
			nlj = nbparm[nlj];
			f14e = charge[atom1]*charge[atom4]/rMag/sqrtf(rMag)/scee[dihType];
			f14v = r6*(12.0f*ljA[nlj]*r6-6.0f*ljB[nlj])/scnb[dihType]/rMag;
			f14v = 0.0f;
			for (k=0;k<nDim;k++) {
				atomicAdd(&f[atom1+k], (f14e+f14v)*r1[k]);
				atomicAdd(&f[atom4+k], -(f14e+f14v)*r1[k]);
			}
		}
		if (atom4 < 0) { atom4 = -atom4;} // atom4 is negative if the torsion is improper
		if (atom3 < 0) { atom3 = -atom3;} // atom3 is negative if we don't want to compute the scaled 1-4 for this torsion

		c11 = 0.0f;
		c22 = 0.0f;
		c33 = 0.0f;
		c12 = 0.0f;
		c13 = 0.0f;
		c23 = 0.0f;
		for (k=0;k<nDim;k++) {
			r1[k] = xyz[atom1+k] - xyz[atom2+k];
			r2[k] = xyz[atom2+k] - xyz[atom3+k];
			r3[k] = xyz[atom3+k] - xyz[atom4+k];
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
			fdih = dihNs[dihType] * dihKs[dihType] * sinf(dihNs[dihType]*phi-dihPs[dihType])/sinf(phi)*c22/b;
		}
		for (k=0;k<3;k++) {
			f1=fdih*(t1*r1[k]+t2*r2[k]+t3*r3[k])/t3;
			f4=-fdih*(t4*r1[k]+t5*r2[k]+t6*r3[k])/t4;
			atomicAdd(&f[atom1+k], f1);
			atomicAdd(&f[atom2+k], -(1.0f+c12/c22)*f1+c23/c22*f4);
			atomicAdd(&f[atom3+k], c12/c22*f1-(1.0f+c23/c22)*f4);
			atomicAdd(&f[atom4+k], f4);
		}

	}
}

/* C wrappers for kernels */

extern "C" void dih_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, int *dihAtoms_d, float *dihKs_d, float *dihNs_d, float *dihPs_d, int nDihs, float *scee_d, float *scnb_d, float *charge_d, float *ljA_d, float *ljB_d, int *atomType_d, int *nbparm_d, int nAtomTypes) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dih_force_kernel, 0, nDihs); 

    	// Round up according to array size 
    	gridSize = (nDihs + blockSize - 1) / blockSize; 
	// run nondih cuda kernel
	//dih_force_kernel<<<gridSize, blockSize, nAtoms*nDim*sizeof(float)>>>(xyz_d, f_d, nAtoms, lbox, dihAtoms_d, dihKs_d, dihNs_d, dihPs_d, nDihs);
	dih_force_kernel<<<gridSize, blockSize>>>(xyz_d, f_d, nAtoms, lbox, dihAtoms_d, dihKs_d, dihNs_d, dihPs_d, nDihs, scee_d, scnb_d, charge_d, ljA_d, ljB_d, atomType_d, nbparm_d, nAtomTypes);

}

