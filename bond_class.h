
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "constants.h"
//#include "bond_force_cuda.h"

class bond
{
	public:
		int nBonds;
		int nBondHs;
		int nBondnHs;
		int nTypes;
		float *bondKUnique;
		float *bondX0Unique;
		int *bondAtoms_h;
		float *bondKs_h;
		float *bondX0s_h;
		int *bondAtoms_d;
		float *bondKs_d;
		float *bondX0s_d;
		// gpu kernel call size stuff
		int gridSize;
		int blockSize;
		int minGridSize;

		void allocate();
		void initialize_gpu();
		void free_arrays();
		void free_arrays_gpu();

};
