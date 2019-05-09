
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
		float *dihKUnique;
		float *dihNUnique;
		float *dihPUnique;
		int *dihAtoms_h;
		float *dihKs_h;
		float *dihNs_h;
		float *dihPs_h;
		int *dihAtoms_d;
		float *dihKs_d;
		float *dihNs_d;
		float *dihPs_d;

		// scaled NB interactions to be computed in dihedral routine
		float *sceeScaleFactor_h;
		float *sceeScaleFactor_d;
		float *scnbScaleFactor_h;
		float *scnbScaleFactor_d;


		void allocate();
		void initialize_gpu();
		void free_arrays();
		void free_arrays_gpu();

};
