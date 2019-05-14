
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "constants.h"
#include "init_rand.cuh"

class atom
{
	private:
		int i, j, k;
		FILE *forceXyzFile;
		FILE *xyzFile;
		FILE *vFile;
		float sigma;
		float rand_gauss();
	public:
		int nAtoms;
		int nTypes;
		int excludedAtomsListLength;
		int nAtomBytes;
		int nTypeBytes;
		int nMols;	// number of molecules
		int *molPointer_h;
		int *molPointer_d;
		float *xyz_h;    // coordinate array - host data
		float *xyz_d;    // coordinate array - device data
		float *f_h;      // force array - host data
		float *f_d;      // force array - device data
		float *v_h;      // velocity array - host data
		float *v_d;      // velocity array - device data
		float *mass_h;      // mass array - host data
		float *mass_d;      // mass array - device data
		int *ityp_h;     // atom type array - host data
		int *ityp_d;     // atom type array - device data
		int *nExcludedAtoms_h;     // number of excluded atoms per atom array - host data
		int *nExcludedAtoms_d;     // number of excluded atoms per atom array - device data
		int *excludedAtomsList_h;     // excluded atoms list - host data
		int *excludedAtomsList_d;     // excluded atoms list - device data
		int *nonBondedParmIndex_h;     // nonbonded parameter index - host data
		int *nonBondedParmIndex_d;     // nonbonded parameter index - device data
		float *x0_h;     // center position of parabola and g - host data 
		float *x0_d;     // center position of parabola and g - device data
		float *g0_h;     // height of parabola approximation of g - host data 
		float *g0_d;     // height of parabola approximation of g - device data
		float *gr2_h;     // excluded volume distance and end of parabola distance squared - host data 
		float *gr2_d;     // excluded volume distance and end of parabola distance squared - device data
		float *w_h;      // width of parabola - host data
		float *w_d;      // width of parabola - device data
		float *alpha_h;  // alpha parameter for g - host data
		float *alpha_d;  // alpha parameter for g - device data
		float *vtot_h;  // Monte Carlo normalization factor - host data
		float *vtot_d;  // Monte Carlo normalization factor - device data
		float *charges_h;    // coordinate array - host data
		float *charges_d;    // coordinate array - device data
		float *ljA_h;   // Lennard-Jones A parameter - host data
		float *ljA_d;   // Lennard-Jones A parameter - device data
		float *ljB_h;   // Lennard-Jones B parameter - host data
		float *ljB_d;   // Lennard-Jones B parameter - device data
		int numNNmax;
		int *numNN_d;   // list of neighors per atom
		int *numNN_h;   // list of neighors per atom
		int *NN_d;       // neighbor list - will be size nAtoms*nNNmax
		int *NN_h;       // neighbor list - will be size nAtoms*nNNmax
		int   *key;      // array to determine current position of atoms (after shuffle)
		// nAtoms kernel grid/block configurations
		int gridSize;
		int blockSize;
		int minGridSize;
		// random number generator on gpu
		curandState *randStates_d;
		
		// allocate arrays
		void allocate();
		void allocate_molecule_arrays();
		// read initial coordinates from rst file
		void read_initial_positions(char *);
		// initialize all atom velocities and solvent parameters
		void initialize(float T, float lbox, int nMC);
		// initialize all arrays on GPU memory
		void initialize_gpu(int);
		// copy parameter arrays to GPU
		void copy_params_to_gpu();
		// copy position, force and velocity arrays to GPU
		void copy_pos_v_to_gpu();
		// copy position, force, and velocity arrays from GPU
		void get_pos_f_v_from_gpu();
		// copy position, and velocity arrays from GPU
		void get_pos_v_from_gpu();
		// print positions
		void print_xyz();
		// print velocities
		void print_v();
		// print forces
		void print_forces();
		// reorder
		void reorder();
		// free all arrays on CPU memory
		void free_arrays();
		// free all arrays on GPU memory
		void free_arrays_gpu();

		
};
