
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include "constants.h"

using namespace std;
#include "atom_class.h"

void atom::allocate()
{
	int count, i, j;
	// atoms and types
	numNNmax = 576;
	// size of pos arrays
	nAtomBytes = nAtoms*sizeof(float);
	nTypeBytes = nTypes*sizeof(float);
	// allocate atom coordinate arrays
	cudaMallocHost((float4 **) &pos_h, nAtoms*sizeof(float4));
	// allocate atom velocity arrays
	cudaMallocHost((float4 **) &vel_h, nAtoms*sizeof(float4));
	// allocate atom mass arrays
	cudaMallocHost((float4 **) &mass_h, nAtoms*sizeof(float4));
	// allocate atom force arrays
	cudaMallocHost((float4 **) &for_h, nAtoms*sizeof(float4));
	// allocate atom force arrays
	cudaMallocHost((float4 **) &isspaf_h, nAtoms*sizeof(float4));
	// allocate atom type arrays
	ityp_h = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom type arrays
	nExcludedAtoms_h = (int *)malloc(nAtoms*sizeof(int));
	excludedAtomsList_h = (int *)malloc(excludedAtomsListLength*sizeof(int));
	// allocate atom type arrays
	nonBondedParmIndex_h = (int *)malloc(nTypes*nTypes*sizeof(int));
	//neighborCount_h = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom based parameter arrays
	lj_h = (float2 *)malloc(nTypes*(nTypes+1)/2*sizeof(float2));
	// debug
	nPairs = nAtoms*nAtoms;
}

void atom::allocate_molecule_arrays()
{
	// allocate molecule pointer array
	molPointer_h = (int *)malloc(nMols*sizeof(int));

}
	
void atom::initialize_velocities(float T)
{
	// initialize velocities
	for (i=0;i<nAtoms;i++) {
		//vel_h[i] = make_float4(rand_gauss()*sqrt(T/mass_h[i]),rand_gauss()*sqrt(T/mass_h[i]),rand_gauss()*sqrt(T/mass_h[i]),mass_h[i]);	
		vel_h[i].x = rand_gauss()*sqrt(T/vel_h[i].w);
		vel_h[i].y = rand_gauss()*sqrt(T/vel_h[i].w);
		vel_h[i].z = rand_gauss()*sqrt(T/vel_h[i].w);
	}


}

void atom::open_traj_files(char *forOutFileName, char *posOutFileName, char *velOutFileName) 
{
	// open files for printing later
	forFile = fopen(forOutFileName,"w");
	posFile = fopen(posOutFileName,"w");
	velFile = fopen(velOutFileName,"w");
	IFFile = fopen("ISSPA_force.xyz","w");

}

void atom::read_initial_positions(char *inputFileName) {

	char line[MAXCHAR];
	char temp[13];
	char *bunk;
	FILE *coordFile = fopen(inputFileName, "r");
	int nLines, i, j;
	float tempPos[nAtoms*3];

	if ( coordFile != NULL) {
		/* skip first two line */
		bunk = fgets(line, MAXCHAR, coordFile);
		bunk = fgets(line, MAXCHAR, coordFile);
		/* loop over atom position lines */
		nLines = (int) ( nAtoms / 2 ); 
		for (i=0;i<nLines;i++) {
			bunk = fgets(line, MAXCHAR, coordFile);
			for (j=0;j<6;j++) {
				tempPos[i*6+j] = atof(strncpy(temp,line+j*12,12));
			}
		}
		if (nAtoms%2 != 0) {
			bunk = fgets(line, MAXCHAR, coordFile);
			for (j=0;j<3;j++) {
				tempPos[nLines*6+j] = atof(strncpy(temp,line+j*12,12));
			}

		}
		// now populate pos_h float4 array
		for (i=0;i<nAtoms;i++) {
			pos_h[i].x = tempPos[i*3];
			pos_h[i].y = tempPos[i*3+1];
			pos_h[i].z = tempPos[i*3+2];
		}
	} else {
		printf("Could not find coordinate file\n");
		exit(1);
	}

}

void atom::read_initial_velocities(char *inputFileName) {

	char line[MAXCHAR];
	char temp[13];
	char *bunk;
	FILE *velFile = fopen(inputFileName, "r");
	int nLines, i, j;
	float tempVel[nAtoms*3];

	if ( velFile != NULL) {
		/* skip first two line */
		bunk = fgets(line, MAXCHAR, velFile);
		bunk = fgets(line, MAXCHAR, velFile);
		/* loop over atom position lines */
		nLines = (int) ( nAtoms / 2 ); 
		for (i=0;i<nLines;i++) {
			bunk = fgets(line, MAXCHAR, velFile);
			for (j=0;j<6;j++) {
				tempVel[i*6+j] = atof(strncpy(temp,line+j*12,12));
			}
		}
		if (nAtoms%2 != 0) {
			bunk = fgets(line, MAXCHAR, velFile);
			for (j=0;j<3;j++) {
				tempVel[nLines*6+j] = atof(strncpy(temp,line+j*12,12));
			}

		}
		// now populate pos_h float4 array
		for (i=0;i<nAtoms;i++) {
			vel_h[i].x = tempVel[i*3];
			vel_h[i].y = tempVel[i*3+1];
			vel_h[i].z = tempVel[i*3+2];
		}
	} else {
		printf("Could not find velocity file\n");
		exit(2);
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

void atom::initialize_gpu(int seed)
{
	// allocate atom coordinate arrays
	cudaMalloc((void **) &pos_d, nAtoms*sizeof(float4));
	// allocate atom velocity arrays
	cudaMalloc((void **) &vel_d, nAtoms*sizeof(float4));
	// allocate atom force arrays
	cudaMalloc((void **) &for_d, nAtoms*sizeof(float4));
	// allocate atom force arrays
	cudaMalloc((void **) &isspaf_d, nAtoms*sizeof(float4));
	// allocate mass array
	//cudaMalloc((void **) &mass_d, nAtomBytes);
	// allocate neighborlist stuff
	//cudaMalloc((void **) &numNN_d, nAtoms*sizeof(int));
	//cudaMalloc((void **) &NN_d, nAtoms*numNNmax*sizeof(int2));
	//cudaMalloc((void **) &neighborList_d, nAtoms*numNNmax*sizeof(int4));
	//cudaMalloc((void **) &neighborCount_d, nAtoms*sizeof(int));
	// allocate atom type arrays
	cudaMalloc((void **) &ityp_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &nonBondedParmIndex_d, nTypes*nTypes*sizeof(int));
	// exluded atoms stuff
	cudaMalloc((void **) &nExcludedAtoms_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &excludedAtomsList_d, excludedAtomsListLength*sizeof(int));
	// allocate atom based parameter arrays
	cudaMalloc((void **) &lj_d, nTypes*(nTypes+1)/2*sizeof(float2));
	// random number states
	cudaMalloc((void**) &randStates_d, nAtoms*sizeof(curandState));
	init_rand_states(randStates_d, seed, nAtoms);
	// timing
	cudaEventCreate(&nonbondStart);
	cudaEventCreate(&nonbondStop);
	cudaEventCreate(&neighborListStart);
	cudaEventCreate(&neighborListStop);
	cudaEventCreate(&leapFrogStart);
	cudaEventCreate(&leapFrogStop);

}	


// copy parameter arrays to GPU
void atom::copy_params_to_gpu() {

	// copy data to device
	cudaMemcpy(ityp_d, ityp_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nonBondedParmIndex_d, nonBondedParmIndex_h, nTypes*nTypes*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nExcludedAtoms_d, nExcludedAtoms_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(excludedAtomsList_d, excludedAtomsList_h, excludedAtomsListLength*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(lj_d, lj_h, nTypes*(nTypes+1)/2*sizeof(float2), cudaMemcpyHostToDevice);	
}
// copy position, force and velocity arrays to GPU
void atom::copy_pos_vel_to_gpu() {
	cudaMemcpy(vel_d, vel_h, nAtoms*sizeof(float4), cudaMemcpyHostToDevice);	
	cudaMemcpy(pos_d, pos_h, nAtoms*sizeof(float4), cudaMemcpyHostToDevice);	
}
// copy position, force, and velocity arrays from GPU
void atom::get_pos_vel_for_from_gpu() {
	// pass device variable, f_d, to host variable f_h
	cudaMemcpy(for_h, for_d, nAtoms*sizeof(float4), cudaMemcpyDeviceToHost);	
	// pass device variable, pos_d, to host variable pos_h
	cudaMemcpy(pos_h, pos_d, nAtoms*sizeof(float4), cudaMemcpyDeviceToHost);	
	// pass device variable, v_d, to host variable v_h
	cudaMemcpy(vel_h, vel_d, nAtoms*sizeof(float4), cudaMemcpyDeviceToHost);	
	// pass device variable, v_d, to host variable v_h
	cudaMemcpy(isspaf_h, isspaf_d, nAtoms*sizeof(float4), cudaMemcpyDeviceToHost);	
}
// copy position, and velocity arrays from GPU
void atom::get_pos_vel_from_gpu() {
	// grab velocities from device
	cudaMemcpy(vel_h, vel_d, nAtoms*sizeof(float4), cudaMemcpyDeviceToHost);	
	// grab positions from device
	cudaMemcpy(pos_h, pos_d, nAtoms*sizeof(float4), cudaMemcpyDeviceToHost);	
}

void atom::print_for() {
	int ip;
	fprintf(forFile,"%d\n", nAtoms);
	fprintf(forFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		fprintf(forFile,"C %10.6f %10.6f %10.6f\n", for_h[i].x,for_h[i].y,for_h[i].z);
	}
	fflush(forFile);
}

void atom::print_isspaf() {
	int ip;
	fprintf(IFFile,"%d\n", nAtoms);
	fprintf(IFFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
	  fprintf(IFFile,"C %10.6f %10.6f %10.6f %16.8e %16.8e %16.8e\n", pos_h[i].x, pos_h[i].y, pos_h[i].z, isspaf_h[i].x, isspaf_h[i].y, isspaf_h[i].z);
	}
	fflush(IFFile);
}

void atom::print_pos() {
	int ip;
	fprintf(posFile,"%d\n", nAtoms);
	fprintf(posFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		fprintf(posFile,"C %10.6f %10.6f %10.6f\n", pos_h[i].x, pos_h[i].y, pos_h[i].z);
	}
	fflush(posFile);
}

void atom::print_vel() {
	int ip;
	fprintf(velFile,"%d\n", nAtoms);
	fprintf(velFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		fprintf(velFile,"C %10.6f %10.6f %10.6f\n", vel_h[i].x, vel_h[i].y, vel_h[i].z);
	}
	fflush(velFile);
}

void atom::write_rst_files(char *posRstFileName, char *velRstFileName, float lbox) {
	int ip;
	FILE *posRstFile;
	FILE *velRstFile;
	// make sure position and velocities are up-to-date
	get_pos_vel_from_gpu();
	// open position restart file and write
	posRstFile = fopen(posRstFileName, "w");
	fprintf(posRstFile,"Restart Positions from ISSPA CUDA Code\n");
	fprintf(posRstFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i+=2) 
	{
		fprintf(posRstFile,"%12.7f%12.7f%12.7f", pos_h[i].x, pos_h[i].y, pos_h[i].z);
		if(i+1<nAtoms) {
			fprintf(posRstFile,"%12.7f%12.7f%12.7f\n", pos_h[i+1].x, pos_h[i+1].y, pos_h[i+1].z);
		} else { 
			fprintf(posRstFile, "\n");
		}
	}
	fprintf(posRstFile,"%12.7f%12.7f%12.7f%12.7f%12.7f%12.7f\n",lbox,lbox,lbox,90.0,90.0,90.0);
	fclose(posRstFile);
	// open velocity restart file and write
	velRstFile = fopen(velRstFileName, "w");
	fprintf(velRstFile,"Restart Velocities from ISSPA CUDA Code\n");
	fprintf(velRstFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i+=2) 
	{
		fprintf(velRstFile,"%12.7f%12.7f%12.7f", vel_h[i].x, vel_h[i].y, vel_h[i].z);
		if(i+1<nAtoms) {
			fprintf(velRstFile,"%12.7f%12.7f%12.7f\n", vel_h[i+1].x, vel_h[i+1].y, vel_h[i+1].z);
		} else { 
			fprintf(velRstFile, "\n");
		}
	}
	fprintf(velRstFile,"%12.7f%12.7f%12.7f%12.7f%12.7f%12.7f\n",lbox,lbox,lbox,90.0,90.0,90.0);
	fclose(velRstFile);
}
	
void atom::free_arrays() {
	// free host variables
	cudaFree(pos_h);
	cudaFree(vel_h);
	cudaFree(for_h); 
	cudaFree(isspaf_h); 
	free(ityp_h); 
	free(nExcludedAtoms_h);
	free(excludedAtomsList_h);
	free(nonBondedParmIndex_h);
	free(lj_h); 
	fclose(forFile);
	fclose(posFile);
	fclose(velFile);
}

void atom::free_arrays_gpu() {
	// free device variables
	cudaFree(pos_d); 
	cudaFree(vel_d); 
	cudaFree(for_d); 
	cudaFree(isspaf_d); 
	cudaFree(ityp_d); 
	cudaFree(nonBondedParmIndex_d); 
	cudaFree(nExcludedAtoms_d);
	cudaFree(excludedAtomsList_d);
	cudaFree(lj_d); 
	//cudaFree(neighborList_d);
	//cudaFree(neighborCount_d);
}
