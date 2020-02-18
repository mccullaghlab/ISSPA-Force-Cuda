
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
		int *isspaTypes_h;
		int *isspaTypes_d;
		float4 *mcpos_d;
		//float4 *lj_vtot_h;     // isspa LJ parameter
		//float4 *lj_vtot_d;     // isspa LJ parameter -- device
		float4 *x0_w_vtot_h;
		float4 *x0_w_vtot_d;
		float4 *gr2_g0_alpha_h;
		float4 *gr2_g0_alpha_d;
		//float2 *lj_h;     // isspa LJ parameter
		float *x0_h;     // center position of parabola and g - host data 
		float *x0_d;     // center position of parabola and g - device data 
		float *g0_h;     // height of parabola approximation of g - host data 
		float *alpha_h;  // alpha parameter for g - host data
		float2 *gr2_h;     // excluded volume distance and end of parabola distance squared - host data 
		float *w_h;      // width of parabola - host data
		float *vtot_h;  // Monte Carlo normalization factor - host data
		float *vtot_d;  // Monte Carlo normalization factor - device data
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


