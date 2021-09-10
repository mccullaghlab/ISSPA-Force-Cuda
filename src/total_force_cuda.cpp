#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "atom_class.h"
#include "bond_class.h"
#include "angle_class.h"
#include "dih_class.h"
#include "isspa_class.h"
#include "us_class.h"
#include "config_class.h"
#include "isspa_force_cuda.h"
#include "us_force_cuda.h"
#include "nonbond_force_cuda.h"
#include "bond_force_cuda.h"
#include "angle_force_cuda.h"
#include "dih_force_cuda.h"
#include "leapfrog_cuda.h"
#include "neighborlist_cuda.h"
#include "timing_class.h"
#include "read_prmtop.h"
#include "constants.h"
//#include "constants_cuda.cuh"

using namespace std;

int main(int argc, char* argv[])
{
	timing times;
	atom atoms;
	bond bonds;
	angle angles;
	dih dihs;
	config configs;
	isspa isspas;
	us bias;
	int i, j;
	int step;
	int device;
	cudaDeviceProp prop;

	
	cudaGetDevice(&device);
	cudaSetDevice(device);
	printf("Currently using device:%d\n",device);
	cudaGetDeviceProperties(&prop,device);
	printf("GPU Device name: %s\n",prop.name);
	printf("Device global memory: %zu\n",prop.totalGlobalMem);
	printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);
	printf("Warp size: %d\n",prop.warpSize);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max grid size: (%d,%d,%d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
	
	// read config file
	configs.initialize(argv[1]);
	// read atom parameters
	read_prmtop(configs.prmtopFileName, atoms, bonds, angles, dihs, configs.eps);
	// read US parameters if defined
	if (configs.us == 1) {
		printf("Umbrella sampling is turned on.  Reading US parameter file...\n");
		bias.initialize(configs.usCfgFileName);	
		bias.populate_mass(atoms.mass_h,atoms.nAtoms);
		bias.initialize_gpu();
		us_grid_block(bias);
	}
	// initialize atom positions, velocities and solvent parameters
	atoms.read_initial_positions(configs.inputCoordFileName);
	if (configs.velRst == 1) {
		atoms.read_initial_velocities(configs.inputVelFileName);
	} else {
		atoms.initialize_velocities(configs.T);
	}
	atoms.open_traj_files(configs.forOutFileName, configs.posOutFileName, configs.velOutFileName);

	atoms.initialize_gpu(configs.seed);
	//leapfrog_cuda_grid_block(atoms.nAtoms, &atoms.gridSize, &atoms.blockSize, &atoms.minGridSize);
	nonbond_force_cuda_grid_block(atoms, configs.rCut2, configs.lbox);
	// initialize bonds on gpu
	bonds.initialize_gpu();
	bond_force_cuda_grid_block(bonds.nBonds, &bonds.gridSize, &bonds.blockSize, &bonds.minGridSize);
	// initialize angles on gpu
	angles.initialize_gpu();
	angle_force_cuda_grid_block(angles.nAngles, &angles.gridSize, &angles.blockSize, &angles.minGridSize);
	// initialize dihs on gpu
	dihs.initialize_gpu();
	dih_force_cuda_grid_block(dihs.nDihs, &dihs.gridSize, &dihs.blockSize, &dihs.minGridSize);
	// initialize isspa
	isspas.read_isspa_prmtop(configs.isspaPrmtopFileName, configs.nMC);	
	isspas.initialize_gpu(atoms.nAtoms, configs.seed);
	isspa_grid_block(atoms.nAtoms, atoms.nPairs, configs.lbox, isspas);
	 
	// initialize timing
	times.initialize();
	// copy atom data to device
	times.startWriteTimer();
	atoms.copy_params_to_gpu();
	atoms.copy_pos_vel_to_gpu();
	times.stopWriteTimer();

	for (step=0;step<configs.nSteps;step++) {
                //printf("%d\n", step);
		//if (step%configs.deltaNN==0) {
			// compute the neighborlist
		//	times.neighborListTime += neighborlist_cuda(atoms, configs.rNN2, configs.lbox);
		//}
		// zero force array on gpu
		cudaMemset(atoms.for_d, 0.0f,  atoms.nAtoms*sizeof(float4));
		// compute bond forces on device
		times.bondTime += bond_force_cuda(atoms.pos_d, atoms.for_d, atoms.nAtoms, configs.lbox, bonds);
		
		// compute angle forces on device
		times.angleTime += angle_force_cuda(atoms.pos_d, atoms.for_d, atoms.nAtoms, configs.lbox, angles);

		// compute dihedral forces on device
		times.dihTime += dih_force_cuda(atoms, dihs, configs.lbox);

		// run isspa force cuda kernel
		times.isspaTime += isspa_force_cuda(atoms.pos_d, atoms.for_d, atoms.isspaf_d, isspas, atoms.nAtoms);

		// run nonbond cuda kernel
		times.nonbondTime += nonbond_force_cuda(atoms, isspas, atoms.nAtoms);

		if (configs.us == 1) {
			// run US CUDA kernel
			times.usTime += us_force_cuda(atoms.pos_d, atoms.for_d, bias, configs.lbox, atoms.nAtoms);
		}
		// print stuff every so often
		//		if (step > 0 && step%configs.deltaWrite==0) {
		if (step%configs.deltaWrite==0) {
			times.startWriteTimer();
			// get positions, velocities, and forces from gpu
			atoms.get_pos_vel_for_from_gpu();
			// get isspa force and write to file
			atoms.print_isspaf();
			// print force xyz file
			atoms.print_for();
			// print xyz file
			atoms.print_pos();
			// print v file
			atoms.print_vel();
			if (configs.us == 1) {
				// write CV
				bias.print_cv(step, configs.lbox);
			}
			// stop timer
			times.stopWriteTimer();
		}

		// Move atoms and velocities
		times.leapFrogTime += leapfrog_cuda(atoms, configs);
	}

	// write rst files
	times.startWriteTimer();
	atoms.write_rst_files(configs.posRstFileName, configs.velRstFileName, configs.lbox);
	times.stopWriteTimer();
	// print timing info
	times.print_final(configs.nSteps*configs.dtPs*1.0e-3);

	// free up arrays
	atoms.free_arrays();
	atoms.free_arrays_gpu();
	bonds.free_arrays();
	bonds.free_arrays_gpu();
	angles.free_arrays();
	angles.free_arrays_gpu();
	dihs.free_arrays();
	dihs.free_arrays_gpu();
	isspas.free_arrays();
	isspas.free_arrays_gpu();
	if (configs.us == 1) {
		bias.free_arrays();
		bias.free_arrays_gpu();
	}

}
