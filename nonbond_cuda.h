
//__global__ void isspa_force_kernel(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC);

extern "C" void nonbond_cuda(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float lbox);

