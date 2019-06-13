
//extern "C" void angle_force_cuda(float *xyz_d, float *f_d, int nAtoms, float lbox, int *angleAtoms_d, float *angleKs_d, float *angleX0s_d, int nAngles);
float angle_force_cuda(float4 *xyz_d, float4 *f_d, int nAtoms, float lbox, angle& angles);
extern "C" void angle_force_cuda_grid_block(int nAngles, int *gridSize, int *blockSize, int *minGridSize);
