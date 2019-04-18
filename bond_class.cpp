
#include "bond_class.h"

void bond::allocate()
{
	//
	bondAtoms= (int *)malloc(2*nBonds*sizeof(int));
	bondKs= (float *)malloc(nBonds*sizeof(float));
	bondX0s= (float *)malloc(nBonds*sizeof(float));
	bondKUnique = (float *)malloc(nTypes*sizeof(float));
	bondX0Unique = (float *)malloc(nTypes*sizeof(float));
}
