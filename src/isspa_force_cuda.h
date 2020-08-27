
// determine gridsize and blocksize for kernels at beginning of run
void isspa_grid_block(int nAtoms_h, int nPairs_h, float lbox_h, isspa& isspas);
// call ISSPA kernels
float isspa_force_cuda(float4 *xyz, float4 *f, float4 *isspaf, isspa& isspas, int nAtoms_h);
//float isspa_force_cuda(float4 *xyz, float4 *f, isspa& isspas, int nAtoms_h);

