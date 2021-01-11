
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "constants.h"
#include "init_rand.cuh"

class atom
{
        private:
                int i, j, k;
		FILE *forFile;
		FILE *posFile;
		FILE *velFile;
		FILE *IFFile;
                float sigma;
		float rand_gauss();
	public:
		int nAtoms;
		int nPairs;
		int nTypes;
		int excludedAtomsListLength;
		int nAtomBytes;
		int nTypeBytes;
		int nMols;	// number of molecules
		int *molPointer_h;
		int *molPointer_d;
		float4 *pos_h;    // coordinate array - host data
		float4 *pos_d;    // coordinate array - device data
		float4 *for_h;      // force array - host data
		float4 *for_d;      // force array - device data
		float4 *isspaf_h;      // force array - host data
		float4 *isspaf_d;      // force array - host data
		float4 *vel_h;      // velocity array - host data
		float4 *vel_d;      // velocity array - device data
		float4 *mass_h;      // velocity array - host data
		int *ityp_h;     // atom type array - host data
		int *ityp_d;     // atom type array - device data
		int *nExcludedAtoms_h;     // number of excluded atoms per atom array - host data
		int *nExcludedAtoms_d;     // number of excluded atoms per atom array - device data
		int *excludedAtomsList_h;     // excluded atoms list - host data
		int *excludedAtomsList_d;     // excluded atoms list - device data
		int *nonBondedParmIndex_h;     // nonbonded parameter index - host data
		int *nonBondedParmIndex_d;     // nonbonded parameter index - device data
		float2 *lj_h;
		float2 *lj_d;
		int numNNmax;
		int4 *neighborList_d;
		int *neighborCount_h;  // number of neighbors in list on host
		int *neighborCount_d;  // number of neighbors in list on device
		int totalNeighbors;    // size of neighborlist
		// nAtoms kernel grid/block configurations
		int gridSize;
                int blockSize;
                int minGridSize;
                int nThreads;
                // random number generator on gpu
		curandState *randStates_d;
		// gpu timing events
		cudaEvent_t nonbondStart, nonbondStop;
		cudaEvent_t neighborListStart, neighborListStop;
		cudaEvent_t leapFrogStop, leapFrogStart;
		
		// allocate arrays
		void allocate();
		void allocate_molecule_arrays();
		// read initial coordinates from rst file
		void read_initial_positions(char *);
		// read initial velocities from rst file
		void read_initial_velocities(char *);
		// initialize all atom velocities based on MB dist at temperature T
		void initialize_velocities(float T);
		// open trajectory files
		void open_traj_files(char *forOutFileName, char *posOutFileName, char *velOutFileName);
		// initialize all arrays on GPU memory
		void initialize_gpu(int);
		// copy parameter arrays to GPU
		void copy_params_to_gpu();
		// copy position, force and velocity arrays to GPU
		void copy_pos_vel_to_gpu();
		// copy position, force, and velocity arrays from GPU
		void get_pos_vel_for_from_gpu();
		// copy position, and velocity arrays from GPU
		void get_pos_vel_from_gpu();
		// print position trajectory
		void print_pos();
		// print velocity trajectory
		void print_vel();
		// print force trajectory
		void print_for();
		// print force trajectory
		void print_isspaf();
		// write position and velocity restart files
		void write_rst_files(char *posRstFileName, char *velRstFileName, float lbox);
		// reorder
		void reorder();
		// free all arrays on CPU memory
		void free_arrays();
		// free all arrays on GPU memory
		void free_arrays_gpu();

		
};
