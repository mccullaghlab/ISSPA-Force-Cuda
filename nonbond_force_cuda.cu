
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector_functions.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "nonbond_force_cuda.h"

#define nDim 3

// CUDA Kernels

__global__ void nonbond_force_kernel(float4 *xyz, float4 *f, float *charges, float2 *lj, int *ityp, int nAtoms, float rCut2, float lbox, int *NN, int *numNN, int numNNmax, int *nbparm, int nTypes) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x;
	extern __shared__ float4 xyz_s[];
	int atom1;
	int atom2;
	int it, jt;    // atom type of atom of interest
	float dist2;	
	int i, k;
	int N;
	int start;
	float4 r;
	float r6;
	float fc;
	float flj;
	float hbox;
	float2 ljAB;
	int nlj;
	int chunk;


	// copy positions from global memory to shared memory for each block
	chunk = (int) ( (nAtoms+blockDim.x-1)/blockDim.x);
	for (i=t*chunk;i<(t+1)*chunk;i++) {
		xyz_s[i] = __ldg(xyz+i);
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
				//dist2 = 0.0f;
				r = min_image(xyz_s[atom1] - xyz_s[atom2],lbox,hbox);
				dist2 = r.x*r.x + r.y*r.y + r.z*r.z;
				if (dist2 < rCut2) {
					// get interaction type
					it = __ldg(ityp+atom1);
					jt = __ldg(ityp+atom2);
					nlj = nTypes*(it-1)+jt-1;
					nlj = __ldg(nbparm+nlj);
					ljAB = __ldg(lj+nlj);
					// LJ force
					r6 = powf(dist2,-3.0);
					flj = r6 * (12.0 * ljAB.x * r6 - 6.0 * ljAB.y) / dist2;
					fc = __ldg(charges+atom1)*__ldg(charges+atom2)/dist2/sqrtf(dist2);
					f[atom1].x += (flj+fc)*r.x;
					f[atom1].y += (flj+fc)*r.y;
					f[atom1].z += (flj+fc)*r.z;
				}
			}
		}

	}
}

/* C wrappers for kernels */

float nonbond_force_cuda(atom &atoms, float rCut2, float lbox) 
{
	float milliseconds;

	// timing
	cudaEventRecord(atoms.nonbondStart);

	// run nonbond cuda kernel
	nonbond_force_kernel<<<atoms.gridSize, atoms.blockSize, atoms.nAtoms*sizeof(float4)>>>(atoms.pos_d, atoms.for_d, atoms.charges_d, atoms.lj_d, atoms.ityp_d, atoms.nAtoms, rCut2, lbox, atoms.NN_d, atoms.numNN_d, atoms.numNNmax, atoms.nonBondedParmIndex_d, atoms.nTypes);

	// finish timing
	cudaEventRecord(atoms.nonbondStop);
	cudaEventSynchronize(atoms.nonbondStop);
	cudaEventElapsedTime(&milliseconds, atoms.nonbondStart, atoms.nonbondStop);
	return milliseconds;

}

extern "C" void nonbond_force_cuda_grid_block(int nAtoms, int *gridSize, int *blockSize, int *minGridSize)
{
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, nonbond_force_kernel, 0, nAtoms); 

    	// Round up according to array size 
    	*gridSize = (nAtoms + *blockSize - 1) / *blockSize; 

}
