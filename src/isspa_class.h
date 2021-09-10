
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
                int nTypes;      // number of isspa types
		int nMC;         // number of MC points
		int nRs;         // number of distance values in tabulated forces
                int nGRs;         // number of distance values in tabulated densities
                int nERs;         // number of distance values in tabulated densities
                int *isspaTypes_h;
		int *isspaTypes_d;
		float *rmax_h;     // center position of parabola and g - host data 
		float *rmax_d;     // center position of parabola and g - device data 
		float *vtot_h;  // Monte Carlo normalization factor - host data
		float *vtot_d;  // Monte Carlo normalization factor - device data
		float *isspaForceTable_h; //
		float *isspaForceTable_d; //
		float *isspaForceR_h; //
		float *isspaForceR_d; //
		float *isspaGTable_h; //
                float *isspaGTable_d; //
  		float *isspaGR_h; //
                float *isspaGR_d; //
                float *isspaETable_h; //
                float *isspaETable_d; //
                float *isspaER_h; //
                float *isspaER_d; //
                float2 forceRparams;
                float2 gRparams;
                float2 eRparams;
                float4 *mcpos_d;
		float4 *e0now_d;
		float4 *enow_d;
                float *buffer_mcpos_d;
		float4 *buffer_e0now_d;
		float4 *buffer_enow_d;
                // kernel grid/block configurations
		int mcGridSize;
		int mcBlockSize;
                int mcThreads;
                int fieldGridSize;
		int fieldBlockSize;
                int fieldThreads;
                int forceThreads;
                int forceGridSize;
		int forceBlockSize;
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


