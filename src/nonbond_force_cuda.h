//extern "C" void nonbond_cuda(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float rCut2, float lbox, int *NN, int *numNN, int numNNmax, int *nbparm_d, int nTypes);
float nonbond_force_cuda(atom& atoms, isspa& isspas, int nAtoms_h);
extern "C" void nonbond_force_cuda_grid_block(atom& atoms, float rCut2, float lbox);
