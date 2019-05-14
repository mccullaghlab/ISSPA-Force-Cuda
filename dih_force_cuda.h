
float dih_force_cuda(atom& atoms, dih& dihs, float lbox);
extern "C" void dih_force_cuda_grid_block(int nDihs, int *gridSize, int *blockSize, int *minGridSize);

