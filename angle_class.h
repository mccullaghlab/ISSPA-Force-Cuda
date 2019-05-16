
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
		int4 *angleAtoms_h;  // host array of atoms in a given angle with fourth entry being angle type
		int4 *angleAtoms_d;  // device array of atoms in a given angle with fourth entry being angle type
		float2 *angleParams_h; // host array of K and theta 0 parameters for each angle type
		float2 *angleParams_d; // device array of K and theta 0 parameters for each angle type
		// gpu kernel call size stuff
		int gridSize;
		int blockSize;
		int minGridSize;
		// gpu timing stuff
		cudaEvent_t angleStart, angleStop;
		void allocate();
		void initialize_gpu();
		void free_arrays();
		void free_arrays_gpu();

};
