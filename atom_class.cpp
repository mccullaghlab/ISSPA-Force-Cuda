
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "constants.h"

using namespace std;
#include "atom_class.h"

void atom::allocate()
{
	// atoms and types
	numNNmax = 200;
	// size of xyz arrays
	nAtomBytes = nAtoms*sizeof(float);
	nTypeBytes = nTypes*sizeof(float);
	// allocate atom coordinate arrays
//	xyz_h = (float *)malloc(nAtomBytes*nDim);
	cudaMallocHost((float **) &xyz_h, nAtomBytes*nDim);
	// allocate atom velocity arrays
//	v_h = (float *)malloc(nAtomBytes*nDim);
	cudaMallocHost((float **) &v_h, nAtomBytes*nDim);
	// allocate atom force arrays
//	f_h = (float *)malloc(nAtomBytes*nDim);
	cudaMallocHost((float **) &f_h, nAtomBytes*nDim);
	// alocate mass array
	//mass_h = (float *)malloc(nAtoms*sizeof(float));
	cudaMallocHost((float **) &mass_h, nAtomBytes);
	// alocate charge array
	//charges_h = (float *)malloc(nAtoms*sizeof(float));
	cudaMallocHost((float **) &charges_h, nAtomBytes);
	// allocate key array - atom number
	key = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom type arrays
	ityp_h = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom type arrays
	nExcludedAtoms_h = (int *)malloc(nAtoms*sizeof(int));
	excludedAtomsList_h = (int *)malloc(excludedAtomsListLength*sizeof(int));
	// allocate atom type arrays
	nonBondedParmIndex_h = (int *)malloc(nTypes*nTypes*sizeof(int));
	// allocate atom based parameter arrays
	x0_h = (float *)malloc(nTypeBytes);
	g0_h = (float *)malloc(nTypeBytes);
	gr2_h = (float *)malloc(nTypeBytes*2);
	w_h = (float *)malloc(nTypeBytes);
	alpha_h = (float *)malloc(nTypeBytes);
	vtot_h = (float *)malloc(nTypeBytes);
	ljA_h = (float *)malloc(nTypes*(nTypes+1)/2*sizeof(float));
	ljB_h = (float *)malloc(nTypes*(nTypes+1)/2*sizeof(float));
	// debug
	NN_h = (int *)malloc(nAtoms*numNNmax*sizeof(int));
	numNN_h = (int *)malloc(nAtoms*sizeof(int));
}

void atom::allocate_molecule_arrays()
{
	// allocate molecule pointer array
	molPointer_h = (int *)malloc(nMols*sizeof(int));

}
	
void atom::initialize(float T, float lbox, int nMC)
{
	float dist2, temp;
	float sigma2;
	// populate host arrays
	gr2_h[0] = 11.002;
	gr2_h[1] = 21.478;
	w_h[0] = 0.801;
	g0_h[0] = 1.714; // height of parabola
	x0_h[0] = 4.118;
	alpha_h[0] = 2.674; 
//	ljA_h[0] = 6.669e7;
//	ljB_h[0] = 1.103e4;
	vtot_h[0] = 16.0/3.0*3.1415926535*w_h[0]*g0_h[0]/((float) nMC)*0.0334*1E-2;
	sigma = pow(ljA_h[0]/ljB_h[0],(1.0/6.0));
	sigma2 = sigma*sigma;

	// initialize velocities
	for (i=0;i<nAtoms;i++) {
		f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
		//ityp_h[i] = 0;
		//charges_h[i] = 0.0;
		//mass_h[i] = 12.0;
		// reweight hydrogens
		if (mass_h[i] < 2.0) {
			mass_h[i] = 12.0;
		}
		for (k=0;k<nDim;k++) {
			v_h[i*nDim+k] = rand_gauss()*sqrt(T/mass_h[i]);	
		}
	}

	// open files for printing later
	forceXyzFile = fopen("forces.xyz","w");
	xyzFile = fopen("positions.xyz","w");
	vFile = fopen("velocities.xyz","w");

}

void atom::read_initial_positions(char *inputFileName) {

	char line[MAXCHAR];
	char temp[13];
	char *bunk;
	FILE *coordFile = fopen(inputFileName, "r");
	int nLines, i, j;

	if ( coordFile != NULL) {
		/* skip first two line */
		bunk = fgets(line, MAXCHAR, coordFile);
		bunk = fgets(line, MAXCHAR, coordFile);
		/* loop over atom position lines */
		nLines = (int) ( nAtoms / 2 ); 
		for (i=0;i<nLines;i++) {
			bunk = fgets(line, MAXCHAR, coordFile);
			for (j=0;j<6;j++) {
				xyz_h[i*6+j] = atof(strncpy(temp,line+j*12,12));
			}
		}
		if (nAtoms%2 != 0) {
			bunk = fgets(line, MAXCHAR, coordFile);
			for (j=0;j<3;j++) {
				xyz_h[nLines*6+j] = atof(strncpy(temp,line+j*12,12));
			}

		}
	}

}


float atom::rand_gauss() 
{
	
	float v1, v2, r2, fac;

	v1 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
	v2 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
	r2 = v1*v1 + v2*v2;
	while (r2 > 1.0) {
		v1 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
        	v2 = 1.0 - 2.0 * (float) rand() / (float) RAND_MAX;
		r2 = v1*v1 + v2*v2;
	}
	fac = sqrt(-2.0*log(r2)/r2);
	return v1*fac;
}

void atom::initialize_gpu()
{
	// allocate atom coordinate arrays
	cudaMalloc((void **) &xyz_d, nAtomBytes*nDim);
	// allocate atom velocity arrays
	cudaMalloc((void **) &v_d, nAtomBytes*nDim);
	// allocate atom force arrays
	cudaMalloc((void **) &f_d, nAtomBytes*nDim);
	// allocate mass array
	cudaMalloc((void **) &mass_d, nAtomBytes);
	// allocate charges array
	cudaMalloc((void **) &charges_d, nAtomBytes);
	// allocate neighborlist stuff
	cudaMalloc((void **) &numNN_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &NN_d, nAtoms*numNNmax*sizeof(int));
	// allocate atom type arrays
	cudaMalloc((void **) &ityp_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &nonBondedParmIndex_d, nTypes*nTypes*sizeof(int));
	// exluded atoms stuff
	cudaMalloc((void **) &nExcludedAtoms_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &excludedAtomsList_d, excludedAtomsListLength*sizeof(int));
	// allocate atom based parameter arrays
	cudaMalloc((void **) &x0_d, nTypeBytes);
	cudaMalloc((void **) &g0_d, nTypeBytes);
	cudaMalloc((void **) &gr2_d, nTypeBytes*2);
	cudaMalloc((void **) &w_d, nTypeBytes);
	cudaMalloc((void **) &alpha_d, nTypeBytes);
	cudaMalloc((void **) &vtot_d, nTypeBytes);
	cudaMalloc((void **) &ljA_d, nTypes*(nTypes+1)/2*sizeof(float));
	cudaMalloc((void **) &ljB_d, nTypes*(nTypes+1)/2*sizeof(float));

}	


// copy parameter arrays to GPU
void atom::copy_params_to_gpu() {

	// copy data to device
	cudaMemcpy(ityp_d, ityp_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nonBondedParmIndex_d, nonBondedParmIndex_h, nTypes*nTypes*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nExcludedAtoms_d, nExcludedAtoms_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(excludedAtomsList_d, excludedAtomsList_h, excludedAtomsListLength*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(x0_d, x0_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(g0_d, g0_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(gr2_d, gr2_h, 2*nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(alpha_d, alpha_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(vtot_d, vtot_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(ljA_d, ljA_h, nTypes*(nTypes+1)/2*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(ljB_d, ljB_h, nTypes*(nTypes+1)/2*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(mass_d, mass_h, nAtomBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(charges_d, charges_h, nAtomBytes, cudaMemcpyHostToDevice);	
}
// copy position, force and velocity arrays to GPU
void atom::copy_pos_v_to_gpu() {
	cudaMemcpy(v_d, v_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
	cudaMemcpy(xyz_d, xyz_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
}
// copy position, force, and velocity arrays from GPU
void atom::get_pos_f_v_from_gpu() {
	// pass device variable, f_d, to host variable f_h
	cudaMemcpy(f_h, f_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
	// pass device variable, xyz_d, to host variable xyz_h
	cudaMemcpy(xyz_h, xyz_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
	// pass device variable, v_d, to host variable v_h
	cudaMemcpy(v_h, v_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
}
// copy position, and velocity arrays from GPU
void atom::get_pos_v_from_gpu() {
	// pass device variable, f_d, to host variable f_h
	cudaMemcpy(v_h, v_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
	// pass device variable, xyz_d, to host variable xyz_h
	cudaMemcpy(xyz_h, xyz_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
}

void atom::print_forces() {
	int ip;
	fprintf(forceXyzFile,"%d\n", nAtoms);
	fprintf(forceXyzFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		ip = key[i];
		fprintf(forceXyzFile,"C %10.6f %10.6f %10.6f\n", f_h[i*nDim],f_h[i*nDim+1],f_h[i*nDim+2]);
	}
	fflush(forceXyzFile);
}

void atom::print_xyz() {
	int ip;
	fprintf(xyzFile,"%d\n", nAtoms);
	fprintf(xyzFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		ip = key[i];
		fprintf(xyzFile,"C %10.6f %10.6f %10.6f\n", xyz_h[i*nDim], xyz_h[i*nDim+1], xyz_h[i*nDim+2]);
	}
	fflush(xyzFile);
}

void atom::print_v() {
	int ip;
	fprintf(vFile,"%d\n", nAtoms);
	fprintf(vFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		ip = key[i];
		fprintf(vFile,"C %10.6f %10.6f %10.6f\n", v_h[i*nDim], v_h[i*nDim+1], v_h[i*nDim+2]);
	}
	fflush(vFile);
}

	
void atom::free_arrays() {
	// free host variables
	free(key);
	cudaFree(xyz_h);
	cudaFree(f_h); 
	cudaFree(charges_h); 
	cudaFree(mass_h); 
	free(ityp_h); 
	free(w_h); 
	free(g0_h); 
	free(gr2_h); 
	free(x0_h); 
	free(alpha_h); 
	free(vtot_h); 
	free(ljA_h); 
	free(ljB_h); 
	fclose(forceXyzFile);
	fclose(xyzFile);
	fclose(vFile);
}

void atom::free_arrays_gpu() {
	// free device variables
	cudaFree(xyz_d); 
	cudaFree(f_d); 
	cudaFree(ityp_d); 
	cudaFree(nonBondedParmIndex_d); 
	cudaFree(w_d); 
	cudaFree(g0_d); 
	cudaFree(gr2_d); 
	cudaFree(x0_d); 
	cudaFree(alpha_d); 
	cudaFree(vtot_d); 
	cudaFree(ljA_d); 
	cudaFree(ljB_d); 
	cudaFree(charges_d); 
	cudaFree(numNN_d);
	cudaFree(NN_d);
}
