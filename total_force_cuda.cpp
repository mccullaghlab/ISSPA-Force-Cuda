
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "isspa_force_cuda.h"
#include "nonbond_cuda.h"
#include "bond_force_cuda.h"
#include "angle_force_cuda.h"
#include "dih_force_cuda.h"
#include "leapfrog_cuda.h"
#include "neighborlist_cuda.h"
#include "atom_class.h"
#include "bond_class.h"
#include "angle_class.h"
#include "dih_class.h"
#include "config_class.h"
#include "read_prmtop.h"
#include "constants.h"

using namespace std;

int main(int argc, char* argv[])
{
	cudaEvent_t start, stop;
	float milliseconds;
	float day_per_millisecond;
	atom atoms;
	bond bonds;
	angle angles;
	dih dihs;
	config configs;
	int i;
	int step;
	int *NN_h, *numNN_h;
	long long leapfrog_seed;
	long long isspa_seed;
	isspa_seed = 0;
	leapfrog_seed = 0;

	NN_h = (int *)malloc(atoms.nAtoms*atoms.numNNmax*sizeof(int));
	numNN_h = (int *)malloc(atoms.nAtoms*sizeof(int));


	// read config file
	configs.initialize(argv[1]);
	// read atom parameters
	printf("prmtop file name in main:%s\n",configs.prmtopFileName);
	read_prmtop(configs.prmtopFileName, atoms, bonds, angles, dihs);
//	for (i=0;i<dihs.nDihs;i++) {
//		printf("%4d %4d %4d %4d: %8.3f %8.3f\n", dihs.dihAtoms_h[i*5]/3+1, dihs.dihAtoms_h[i*5+1]/3+1, dihs.dihAtoms_h[i*5+2]/3+1, dihs.dihAtoms_h[i*5+3]/3+1, dihs.sceeScaleFactor_h[dihs.dihAtoms_h[i*5+4]], dihs.scnbScaleFactor_h[dihs.dihAtoms_h[i*5+4]]);
//	}
	// initialize atom positions, velocities and solvent parameters
	atoms.read_initial_positions(configs.inputCoordFileName);
	atoms.initialize(configs.T, configs.lbox, configs.nMC);
	atoms.initialize_gpu();
	// initialize bonds on gpu
	bonds.initialize_gpu();
	// initialize angles on gpu
	angles.initialize_gpu();
	// initialize dihs on gpu
	dihs.initialize_gpu();
	
	// start device timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// copy atom data to device
	atoms.copy_params_to_gpu();
	atoms.copy_pos_v_to_gpu();

	for (step=0;step<configs.nSteps;step++) {

		if (step%configs.deltaNN==0) {
			// compute the neighborlist
			neighborlist_cuda(atoms.xyz_d, atoms.NN_d, atoms.numNN_d, configs.rNN2, atoms.nAtoms, atoms.numNNmax, configs.lbox, atoms.nExcludedAtoms_d, atoms.excludedAtomsList_d);
		}

		// zero force array on gpu
		cudaMemset(atoms.f_d, 0.0f,  atoms.nAtoms*nDim*sizeof(float));

		// compute bond forces on device
		bond_force_cuda(atoms.xyz_d, atoms.f_d, atoms.nAtoms, configs.lbox, bonds.bondAtoms_d, bonds.bondKs_d, bonds.bondX0s_d, bonds.nBonds, bonds.gridSize, bonds.blockSize);
		
		// compute angle forces on device
		angle_force_cuda(atoms.xyz_d, atoms.f_d, atoms.nAtoms, configs.lbox, angles.angleAtoms_d, angles.angleKs_d, angles.angleX0s_d, angles.nAngles);

		// compute dihedral forces on device
		dih_force_cuda(atoms.xyz_d, atoms.f_d, atoms.nAtoms, configs.lbox, dihs.dihAtoms_d, dihs.dihKs_d, dihs.dihNs_d, dihs.dihPs_d, dihs.nDihs, dihs.sceeScaleFactor_d, dihs.scnbScaleFactor_d, atoms.charges_d, atoms.ljA_d, atoms.ljB_d, atoms.ityp_d, atoms.nonBondedParmIndex_d, atoms.nTypes);
		// run isspa force cuda kernal
//		isspa_force_cuda(atoms.xyz_d, atoms.f_d, atoms.w_d, atoms.x0_d, atoms.g0_d, atoms.gr2_d, atoms.alpha_d, atoms.vtot_d, atoms.ljA_d, atoms.ljB_d, atoms.ityp_d, atoms.nAtoms, configs.nMC, configs.lbox, atoms.NN_d, atoms.numNN_d, atoms.numNNmax, isspa_seed);
//		isspa_seed += 1;

		// run nonbond cuda kernel
		nonbond_cuda(atoms.xyz_d, atoms.f_d, atoms.charges_d, atoms.ljA_d, atoms.ljB_d, atoms.ityp_d, atoms.nAtoms, configs.lbox, atoms.NN_d, atoms.numNN_d, atoms.numNNmax, atoms.nonBondedParmIndex_d, atoms.nTypes);

		// print stuff every so often
		if (step%configs.deltaWrite==0) {
			// get positions, velocities, and forces from gpu
			atoms.get_pos_f_v_from_gpu();
			// print force xyz file
			atoms.print_forces();
			// print xyz file
			atoms.print_xyz();
			// print v file
			atoms.print_v();
		}

		// Move atoms and velocities
		leapfrog_cuda(atoms.xyz_d, atoms.v_d, atoms.f_d, atoms.mass_d, configs.T, configs.dt, configs.pnu, atoms.nAtoms, configs.lbox, leapfrog_seed);
		leapfrog_seed += 1;
	}


	// get GPU time
	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time = %20.10f ms\n", milliseconds);
	day_per_millisecond = 1e-3 /60.0/60.0/24.0;
	printf("Average ns/day = %20.10f\n", configs.nSteps*2e-6/(milliseconds*day_per_millisecond) );

	// free up arrays
	atoms.free_arrays();
	atoms.free_arrays_gpu();
	bonds.free_arrays();
	bonds.free_arrays_gpu();
	angles.free_arrays();
	angles.free_arrays_gpu();
	dihs.free_arrays();
	dihs.free_arrays_gpu();

}
