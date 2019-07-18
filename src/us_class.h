
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "constants.h"

class us
{
	public:
		int2 *atomList_h;
		int2 *atomList_d;
		float *mass_h;
		float *mass_d;
		float4 *groupComPos_d;
		int totalBiasAtoms;
		float kumb[2];
		float x0;
		float k;
		float2 groupMass;
		// kernel grid/block configurations
		int usGridSize;
		int usBlockSize;
		// gpu timing events
		cudaEvent_t usStart, usStop;
		void initialize(char *);
		// allocate arrays
		void populate_mass(float4 *, int);
		// initialize all arrays on GPU memory
		void initialize_gpu();
		// clear arrays
		void free_arrays();
		void free_arrays_gpu();

};


