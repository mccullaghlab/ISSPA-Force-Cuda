#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define nDim 3
#define MC 10
//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

void isspa_force(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC);


