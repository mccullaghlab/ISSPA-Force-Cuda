
extern "C" void nonbond_cuda(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float rCut2, float lbox, int *NN, int *numNN, int numNNmax, int *nbparm_d, int nTypes);

