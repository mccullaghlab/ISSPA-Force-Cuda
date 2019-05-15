
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

//#include "bond_class.h"
//extern "C" float bond_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, int *bondAtoms_d, float *bondKs_d, float *bondX0s_d, int nBonds, int gridSize, int blockSize);
float bond_force_cuda(float4 *xyz_d, float4 *f_d, int nAtoms, float lbox, bond& bonds);

extern "C" void bond_force_cuda_grid_block(int nBonds, int *gridSize, int *blockSize, int *minGridSize);

