
// determine gridsize and blocksize for kernels at beginning of run
void us_grid_block(us& bias);
// call US kernel
float us_force_cuda(float4 *xyz, float4 *f, us& bias, float lbox, int nAtoms);

