
#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <math.h>

#define nDim 3

class bond
{
	public:
		int nBonds;
		int nBondHs;
		int nBondnHs;
		int nTypes;
		float *bondKUnique;
		float *bondX0Unique;
		int *bondAtoms;
		float *bondKs;
		float *bondX0s;

		void allocate();

};
