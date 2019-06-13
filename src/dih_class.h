
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "constants.h"

class dih
{
	public:
		int nDihs;
		int nDihHs;
		int nDihnHs;
		int nTypes;
		int4 *dihAtoms_h;
		int4 *dihAtoms_d;
		int *dihTypes_h;
		int *dihTypes_d;
		float4 *dihParams_h;
		float4 *dihParams_d;
		// scaled NB interactions to be computed in dihedral routine
		float2 *scaled14Factors_h;
		float2 *scaled14Factors_d;
		// gpu blocksize info
		int gridSize;
		int blockSize;
		int minGridSize;
		// gpu timing events
		cudaEvent_t dihStart, dihStop;

		void allocate();
		void initialize_gpu();
		void free_arrays();
		void free_arrays_gpu();

};
