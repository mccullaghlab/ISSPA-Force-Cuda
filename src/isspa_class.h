
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "constants.h"
#include "init_rand.cuh"

class isspa
{
	public:
		//DEBUG
		FILE *denFile;
		FILE *forFile;
		int nTypes;      // number of isspa types
		int nMC;         // number of MC points
		int nForceRs;         // number of distance values in tabulated forces
		int nGRs;         // number of distance values in tabulated densities
		float mu;
		float rho;
		int *isspaTypes_h;
		int *isspaTypes_d;
		float4 *mcPos_d;  // Monte Carlo point positions
		float4 *mcFor_d;  // Monte Carlo point forces
		float4 *mcDist_h; // min, max, delta, and normalization of Monte Carlo distribution - host data
		float4 *mcDist_d; // min, max delta, and normalization of Monte Carlo distribution - device data
		float2 *isspaGTable_h; //
		float2 *isspaGTable_d; //
		float *isspaGR_h; //
		float *isspaGR_d; //
		float2 gRparams;
		float *isspaForceTable_h; //
		float *isspaForceTable_d; //
		float *isspaForceR_h; //
		float *isspaForceR_d; //
		float2 forceRparams;
		// kernel grid/block configurations
		int mcGridSize;
		int mcBlockSize;
		int gGridSize;
		int gBlockSize;
		// gpu timing events
		cudaEvent_t isspaStart, isspaStop;
		// random number generator on gpu
		curandState *randStates_d;
		// read isspa prmtop file
		void read_isspa_prmtop(char* isspaPrmtopFileName, int configMC);
		// allocate arrays
		void allocate(int nAtoms, int configMC);
		void construct_parameter_arrays();
		// initialize all arrays on GPU memory
		void initialize_gpu(int nAtoms, int seed);
		// clear arrays
		void free_arrays();
		void free_arrays_gpu();

};


