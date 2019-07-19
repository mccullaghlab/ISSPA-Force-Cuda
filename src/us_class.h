
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "constants.h"

class us
{
	private:
		FILE *cvFile;
	public:
		int2 *atomList_h;
		int2 *atomList_d;
		float *mass_h;
		float *mass_d;
		float4 *groupComPos_d;
		float4 *groupComPos_h;
		int totalBiasAtoms;
		float kumb_h[2];
		float *kumb_d;
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
		// print CV value
		void print_cv(int , float );
		// clear arrays
		void free_arrays();
		void free_arrays_gpu();

};


