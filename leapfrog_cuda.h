

//extern "C" void leapfrog_cuda(float *xyz, float *v, float *f, float *mass, float T, float dt, float pnu, int nAtoms, float lbox, curandState *randStates_d);
float leapfrog_cuda(atom& atoms, config& configs);

