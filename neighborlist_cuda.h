

//extern "C" float neighborlist_cuda(float *xyz, int *NN, int *numNN, float rNN2, int nAtoms, int numNNmax, float lbox, int *nExcludedAtoms, int *excludedAtomsList, int excludedAtomsListLength);
float neighborlist_cuda(atom& atoms, float rNN2, float lbox);

