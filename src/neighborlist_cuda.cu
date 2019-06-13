
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_vector_routines.h"
#include "atom_class.h"
#include "neighborlist_cuda.h"

// CUDA Kernels

__global__ void neighborlist_kernel(float4 *xyz, int4 *neighborList, int *neighborCount, float rNN2, int nAtoms, float lbox, int *nExcludedAtoms, int *excludedAtomsList, int excludedAtomsListLength, int *nbparm, int *ityp, int nTypes, int nPairs, int maxNeighborsPerAtom) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int t = threadIdx.x ;
	extern __shared__ int excludedAtomsList_s[];
	int atom1;
	int atom2;
	float4 temp;
	float dist2;	
	int i;
	int count;
	int exStart;
	int exStop;
	int exPass;
	int exAtom;
	int start;
	int it, jt;    // atom type of atom of interest
	int nlj;
	float hbox;

	// copy excluded atoms list to shared memory
	for (i=t;i<excludedAtomsListLength;i+=blockDim.x) {
		excludedAtomsList_s[i] = __ldg(excludedAtomsList+i);
	}
	__syncthreads();
	// move on

	if (index < nPairs)
	{
		hbox = lbox/2.0;
		atom1 = (int) (index/nAtoms);
		atom2 = index % nAtoms;
		// check exclusions
		exPass = 0;
		if (atom1 < atom2) {
			if (atom1==0) {
				exStart = 0;
			} else {
				//exStart = nExcludedAtoms[atom1-1];
				exStart = __ldg(nExcludedAtoms+atom1-1);
			}
			exStop = __ldg(nExcludedAtoms+atom1);
			for (exAtom=exStart;exAtom<exStop;exAtom++) {
				if (excludedAtomsList_s[exAtom] == atom2) {
					exPass = 1;
					break;
				}
				// the following only applies if exluded atom list is in strictly ascending order
				if (excludedAtomsList_s[exAtom] > atom2) {
					break;
				}
			}
		} else if (atom2 < atom1) {
			if (atom2==0) {
				exStart = 0;
			} else {
				exStart = __ldg(nExcludedAtoms+atom2-1);
			}
			exStop = __ldg(nExcludedAtoms+atom2);
			for (exAtom=exStart;exAtom<exStop;exAtom++) {
				if (excludedAtomsList_s[exAtom] == atom1) {
					exPass = 1;
					break;
				}
				// the following only applies if exluded atom list is in strictly ascending order
				if (excludedAtomsList_s[exAtom] > atom1) {
					break;
				}
			}
		}
		// finish exclusion check
		if (atom1 != atom2 && exPass == 0) {
			// compute distance
			temp = min_image(__ldg(xyz+atom1) - __ldg(xyz+atom2),lbox,hbox);
			dist2 = temp.x*temp.x + temp.y*temp.y + temp.z*temp.z;
			if (dist2 < rNN2) {
				start = maxNeighborsPerAtom*atom1;
				count = atomicAdd(&neighborCount[atom1],1);
				it = __ldg(ityp+atom1);
				jt = __ldg(ityp+atom2);
				nlj = nTypes*(it)+jt;
				nlj = __ldg(nbparm+nlj);
				// set neighborlist in global memory
				neighborList[start + count].x = atom1;
				neighborList[start + count].y = atom2;
				neighborList[start + count].z = nlj;
			}
		}
/*		} else if (atom2 < atom1) {
			// check exclusions
			exPass = 0;
			if (atom2==0) {
				exStart = 0;
			} else {
				exStart = __ldg(nExcludedAtoms+atom2-1);
			}
			exStop = __ldg(nExcludedAtoms+atom2);
			for (exAtom=exStart;exAtom<exStop;exAtom++) {
				if (excludedAtomsList_s[exAtom] == atom1) {
					exPass = 1;
					break;
				}
				// the following only applies if exluded atom list is in strictly ascending order
				if (excludedAtomsList_s[exAtom] > atom1) {
					break;
				}
			}
		}
		// finish exclusion check
		if (atom1 != atom2 && exPass == 0) {
			// compute distance
			temp = min_image(__ldg(xyz+atom1) - __ldg(xyz+atom2),lbox,hbox);
			dist2 = temp.x*temp.x + temp.y*temp.y + temp.z*temp.z;
			if (dist2 < rNN2) {
				count = atomicAdd(neighborCount,1);
				it = __ldg(ityp+atom1);
				jt = __ldg(ityp+atom2);
				nlj = nTypes*(it-1)+jt-1;
				nlj = __ldg(nbparm+nlj);
				// set neighborlist in global memory
				neighborList[count].x = atom1;
				neighborList[count].y = atom2;
				neighborList[count].z = nlj;
			}
		}
		*/
	}
}

/* C wrappers for kernels */

float neighborlist_cuda(atom& atoms, float rNN2, float lbox)
{
	int gridSize;
	int blockSize;
	int minGridSize;
	float milliseconds;
	int i;
	// initialize cuda timing events
	cudaEventRecord(atoms.neighborListStart);
	//cudaMemset(atoms.neighborCount,0, sizeof(int));
	// determine gridSize and blockSize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, neighborlist_kernel, 0, atoms.nPairs); 
    	// Round up according to array size 
    	gridSize = (atoms.nPairs + blockSize - 1) / blockSize; 
	// zero neighborCount
	cudaMemset(atoms.neighborCount_d,0, atoms.nAtoms*sizeof(int));
	// run nonbond cuda kernel
	neighborlist_kernel<<<gridSize, blockSize, atoms.excludedAtomsListLength*sizeof(int)>>>(atoms.pos_d, atoms.neighborList_d, atoms.neighborCount_d, rNN2, atoms.nAtoms, lbox, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d, atoms.excludedAtomsListLength, atoms.nonBondedParmIndex_d, atoms.ityp_d, atoms.nTypes, atoms.nPairs, atoms.numNNmax);
	// copy neighbor count to host and print
	cudaMemcpy(atoms.neighborCount_h, atoms.neighborCount_d,atoms.nAtoms*sizeof(int),cudaMemcpyDeviceToHost);
	atoms.totalNeighbors = 0;
	for (i=0;i<atoms.nAtoms;i++) {
		//atoms.totalNeighbors += atoms.neighborCount_h[i];
		//printf("%3d %3d\n", i+1, atoms.neighborCount_h[i]);
		if (atoms.neighborCount_h[i] > atoms.numNNmax) {
			printf("%3d %3d\n", i+1, atoms.neighborCount_h[i]);
		//	printf("uh oh\n");
		}
	}
	//printf("Number in neighborlist:%d\n", atoms.neighborCount_h[0]);
	// record kernel timin
	cudaEventRecord(atoms.neighborListStop);
    	cudaEventSynchronize(atoms.neighborListStop);
	cudaEventElapsedTime(&milliseconds, atoms.neighborListStart, atoms.neighborListStop);
	return milliseconds;

}

