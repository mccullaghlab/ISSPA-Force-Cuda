
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "constants.h"

class angle
{
	public:
		int nAngles;
		int nAngleHs;
		int nAnglenHs;
		int nTypes;
		float *angleKUnique;
		float *angleX0Unique;
		int *angleAtoms_h;
		float *angleKs_h;
		float *angleX0s_h;
		int *angleAtoms_d;
		float *angleKs_d;
		float *angleX0s_d;
		// gpu kernel call size stuff
		int gridSize;
		int blockSize;
		int minGridSize;

		void allocate();
		void initialize_gpu();
		void free_arrays();
		void free_arrays_gpu();

};
