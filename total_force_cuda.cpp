
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "isspa_force_cuda.h"
#include "nonbond_cuda.h"
#include "leapfrog_cuda.h"
#include "atom_class.h"
#include "config_class.h"

#define nDim 3
#define MC 10

using namespace std;

int main(void)  
{
	int nMC = MC;    // number of MC points
	cudaEvent_t start, stop;
	float milliseconds;
	float day_per_millisecond;
	atom atoms;
	config configs;
	int i;
	int step;

	// initialize
	configs.initialize();
	atoms.initialize(configs.T, configs.lbox);
	atoms.initialize_gpu();

	// start device timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	atoms.copy_params_to_gpu();
	atoms.copy_pos_v_to_gpu();

	for (step=0;step<configs.nSteps;step++) {
		if (step%configs.deltaWrite==0) {
			// get positions, velocities, and forces from gpu
			atoms.get_pos_f_v_from_gpu();
			// print force xyz file
			atoms.print_forces();
			// print xyz file
			atoms.print_xyz();

		}
		if (step%configs.deltaNN==0) {
			// get positions, velocities, and forces from gpu
			atoms.get_pos_v_from_gpu();
			// reorder based on hilbert curve position
			atoms.reorder();
			// send positions and velocities back
			atoms.copy_pos_v_to_gpu();
		}

		// run isspa force cuda kernal
		isspa_force_cuda(atoms.xyz_d, atoms.f_d, atoms.w_d, atoms.x0_d, atoms.g0_d, atoms.gr2_d, atoms.alpha_d, atoms.lj_A_d, atoms.lj_B_d, atoms.ityp_d, atoms.nAtoms, nMC, configs.lbox);

		// run nonbond cuda kernel
		nonbond_cuda(atoms.xyz_d, atoms.f_d, atoms.charges_d, atoms.lj_A_d, atoms.lj_B_d, atoms.ityp_d, atoms.nAtoms, configs.lbox);

		// Move atoms and velocities
		leapfrog_cuda(atoms.xyz_d, atoms.v_d, atoms.f_d, atoms.mass_d, configs.T, configs.dt, configs.pnu, atoms.nAtoms, configs.lbox);
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

	return 0;

}


