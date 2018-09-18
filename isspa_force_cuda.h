
//__global__ void isspa_force_kernel(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC);

extern "C" void isspa_force_cuda(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *vtot, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC, float lbox, int *NN, int *numNN, int numNNmax, long long seed);
