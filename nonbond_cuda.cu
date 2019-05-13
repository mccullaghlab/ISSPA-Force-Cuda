
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "nonbond_cuda.h"

#define nDim 3

// CUDA Kernels

__global__ void nonbond_kernel(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float rCut2, float lbox, int *NN, int *numNN, int numNNmax, int *nbparm, int nTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float xyz_s[];
	int atom1;
	int atom2;
	int it, jt;    // atom type of atom of interest
	float dist2;	
	int i, k;
	int N;
	int start;
	float r[3];
	float r6;
	float fc;
	float flj;
	float hbox;
	int nlj;
	int chunk;


	// copy positions from global memory to shared memory for each block
	chunk = (int) ( (nAtoms*nDim+blockDim.x-1)/blockDim.x);
	for (i=t*chunk;i<(t+1)*chunk;i++) {
		xyz_s[i] = xyz[i];
	}
	__syncthreads();
	// move on
	if (index < nAtoms)
	{
		hbox = lbox/2.0;
		atom1 = index;
		// start position in neighbor list:
		start = atom1*numNNmax;
		// number of atoms in neighbor list:
		N = __ldg(numNN+atom1);
		for (i=0;i<N;i++) {
			atom2 = __ldg(NN+start+i);
			if (atom2 != atom1) {
				// get interaction type
				it = __ldg(ityp+atom1);
				jt = __ldg(ityp+atom2);
				nlj = nTypes*(it-1)+jt-1;
				nlj = __ldg(nbparm+nlj);
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) {
					//r[k] = __ldg(xyz+atom1*nDim+k) - __ldg(xyz+atom2*nDim+k);
					r[k] = xyz_s[atom1*nDim+k] - xyz_s[atom2*nDim+k];
					if (r[k] > hbox) {
//						r[k] -= (int)(temp/lbox+0.5) * lbox;
						r[k] -= lbox;
					} else if (r[k] < -hbox) {
//						r[k] += (int)(-temp/lbox+0.5) * lbox;
						r[k] += lbox;
					}
					dist2 += r[k]*r[k];
				}
				if (dist2 < rCut2) {
					// LJ force
					r6 = powf(dist2,-3.0);
					flj = r6 * (12.0 * __ldg(lj_A+nlj) * r6 - 6.0 * __ldg(lj_B+nlj)) / dist2;
					fc = __ldg(charges+atom1)*__ldg(charges+atom2)/dist2/sqrtf(dist2);
					f[atom1*nDim] += (flj+fc)*r[0];
					f[atom1*nDim+1] += (flj+fc)*r[1];
					f[atom1*nDim+2] += (flj+fc)*r[2];
				}
			}
		}

	}
}

/* C wrappers for kernels */

extern "C" void nonbond_cuda(float *xyz_d, float *f_d, float *charges_d, float *lj_A_d, float *lj_B_d, int *ityp_d, int nAtoms, float rCut2, float lbox, int *NN_d, int *numNN_d, int numNNmax, int *nbparm_d, int nTypes) {
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 

	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nonbond_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	gridSize = (nAtoms + blockSize - 1) / blockSize; 

	// run nonbond cuda kernel
	nonbond_kernel<<<gridSize, blockSize, nAtoms*nDim*sizeof(float)>>>(xyz_d, f_d, charges_d, lj_A_d, lj_B_d, ityp_d, nAtoms, rCut2, lbox, NN_d, numNN_d, numNNmax, nbparm_d, nTypes);

}

