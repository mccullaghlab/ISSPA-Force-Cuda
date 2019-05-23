
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
		int *isspaTypes_h;
		int *isspaTypes_d;
		float4 *mcpos_d;
		float2 *lj_h;     // isspa LJ parameter
		float2 *lj_d;     // isspa LJ parameter -- device
		float *x0_h;     // center position of parabola and g - host data 
		float *x0_d;     // center position of parabola and g - device data
		float *g0_h;     // height of parabola approximation of g - host data 
		float *g0_d;     // height of parabola approximation of g - device data
		float *gr2_h;     // excluded volume distance and end of parabola distance squared - host data 
		float *gr2_d;     // excluded volume distance and end of parabola distance squared - device data
		float *w_h;      // width of parabola - host data
		float *w_d;      // width of parabola - device data
		float *alpha_h;  // alpha parameter for g - host data
		float *alpha_d;  // alpha parameter for g - device data
		float *vtot_h;  // Monte Carlo normalization factor - host data
		float *vtot_d;  // Monte Carlo normalization factor - device data
		// kernel grid/block configurations
		int mcGridSize;
		int mcBlockSize;
		int gGridSize;
		int gBlockSize;
		// gpu timing events
		cudaEvent_t isspaStart, isspaStop;
		// random number generator on gpu
		curandState *randStates_d;
		// allocate arrays
		void allocate(int nAtoms);
		// initialize all arrays on GPU memory
		void initialize_gpu(int nAtoms, int seed);
		// clear arrays
		void free_arrays();
		void free_arrays_gpu();

};


