
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "stringlib.h"
using namespace std;
#include "us_class.h"


void us::initialize(char *cfgFileName)
{
	char *token;
	char *firstToken;
	char temp[LONGCHAR];
	char listGroup1[LONGCHAR];
	char listGroup2[LONGCHAR];
	char cvFileName[MAXCHAR];
	char const *search = "=";
	char const *comment = "#";
	char const *listSep = ",";
	char line[MAXCHAR]; // maximum number of character per line set to 128
	FILE *inFile = fopen(cfgFileName,"r");
	int nBiasAtoms1;
	int nBiasAtoms2;
	int biasAtom;

	printf("Reading US parameters from file: %s\n", cfgFileName);

	if ( inFile != NULL) {
		while (fgets(line, MAXCHAR, inFile) != NULL) {
			firstToken = strtok(line,comment);
			token = trim(strtok(firstToken, search));
			if (strncmp(token,"k",1)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				k = atof(trim(temp));
				printf("US force constant: %f\n",k);
			} else if (strncmp(token,"x0",2)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				x0 = atof(trim(temp));
				printf("US equilibrium position: %f\n",x0);
			} else if (strncmp(token,"cvFile",6)==0) {
				strncpy(temp,strtok(NULL,search),MAXCHAR);
				strcpy(cvFileName,trim(temp));
				printf("Writing CV values to file: %s\n",cvFileName);
			} else if (strncmp(token,"atomSelect1",11)==0) {
				// store list
				strncpy(listGroup1,strtok(NULL,search),LONGCHAR);
				// count number of atoms in list
				strncpy(temp,listGroup1,LONGCHAR);
				token = strtok(temp,listSep);
				nBiasAtoms1 = 0;
				while (token != NULL) {
					token = strtok(NULL,listSep);
					nBiasAtoms1 ++;
				}
				printf("Number of atoms in bias group 1: %d\n", nBiasAtoms1);
			} else if (strncmp(token,"atomSelect2",11)==0) {
				// store list
				strncpy(listGroup2,strtok(NULL,search),LONGCHAR);
				// count number of atoms in list
				strncpy(temp,listGroup2,LONGCHAR);
				token = strtok(temp,listSep);
				nBiasAtoms2 = 0;
				while (token != NULL) {
					token = strtok(NULL,listSep);
					nBiasAtoms2 ++;
				}
				printf("Number of atoms in bias group 2: %d\n", nBiasAtoms2);
			}
		}
		fclose( inFile );
	}
	// populate atom selection lists etc
	totalBiasAtoms = nBiasAtoms1 + nBiasAtoms2;
	printf("Total number of biased atoms: %d\n", totalBiasAtoms);
	// allocate atomList_h
	cudaMallocHost((int2 **) &atomList_h, totalBiasAtoms*sizeof(int2));
	// parse atom list 1
	token = strtok(listGroup1,listSep);
	biasAtom = 0;
	while (token != NULL) {
		// store value into array
		atomList_h[biasAtom].x = atoi(trim(token))-1;
		atomList_h[biasAtom].y = 0;
		token = strtok(NULL,listSep);
		biasAtom++;
	}	
	// parse atom list 2
	token = strtok(listGroup2,listSep);
	while (token != NULL) {
		// store value into array
		atomList_h[biasAtom].x = atoi(trim(token))-1;
		atomList_h[biasAtom].y = 1;
		token = strtok(NULL,listSep);
		biasAtom++;
	}	
	cudaMallocHost((float4 **) &groupComPos_h, 2*sizeof(float4));

	//open cv file
	cvFile = fopen(cvFileName,"w");

}

void us::print_cv(int step, float lbox) {
	int i;
	float cvVal;
	float x, y, z;
	float hbox = lbox/2.0;
	cudaMemcpy(groupComPos_h, groupComPos_d, 2*sizeof(float4), cudaMemcpyDeviceToHost);
	x = groupComPos_h[0].x - groupComPos_h[1].x;
	if (x > hbox) {
		x -= lbox;
	} else if (x < -hbox) {
		x += lbox;
	}
	y = groupComPos_h[0].y - groupComPos_h[1].y;
	if (y > hbox) {
		y -= lbox;
	} else if (y < -hbox) {
		y += lbox;
	}
	z = groupComPos_h[0].z - groupComPos_h[1].z;
	if (z > hbox) {
		z -= lbox;
	} else if (z < -hbox) {
		z += lbox;
	}
	cvVal = sqrt(x*x + y*y + z*z);
	fprintf(cvFile,"%8d  %8.3f\n", step, cvVal);
	fflush(cvFile);
}

void us::populate_mass(float4 *vel, int nAtoms){

	int biasAtom;
	float totalMass[2];	
	// allocate mass_h
	cudaMallocHost((float **) &mass_h, totalBiasAtoms*sizeof(float));
	
	// read masses from vel.w 
	totalMass[0] = 0.0;
	totalMass[1] = 0.0;
	for (biasAtom=0;biasAtom<totalBiasAtoms;biasAtom++) {
		mass_h[biasAtom] = vel[atomList_h[biasAtom].x].w;
		totalMass[atomList_h[biasAtom].y] += mass_h[biasAtom];
	}
	for (biasAtom=0;biasAtom<totalBiasAtoms;biasAtom++) {
		mass_h[biasAtom] /= totalMass[atomList_h[biasAtom].y];
	}
	kumb_h[0] = -k;
	kumb_h[1] = k;
	printf("Mass of group 1: %f\n", totalMass[0]);
	printf("Mass of group 2: %f\n", totalMass[1]);

}

void us::initialize_gpu() {
	int i;
	// allocate arrays on device
	cudaMalloc((void **) &mass_d, totalBiasAtoms*sizeof(float));
	cudaMalloc((void **) &atomList_d, totalBiasAtoms*sizeof(int2));
	cudaMalloc((void **) &groupComPos_d, 2*sizeof(float4));
	cudaMalloc((void **) &kumb_d, 2*sizeof(float));
	// copy data to device
	cudaMemcpy(mass_d, mass_h, totalBiasAtoms*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(atomList_d, atomList_h, totalBiasAtoms*sizeof(int2), cudaMemcpyHostToDevice);
	cudaMemcpy(kumb_d, kumb_h, 2*sizeof(float), cudaMemcpyHostToDevice);
	// initialize timing stuff
	cudaEventCreate(&usStart);
	cudaEventCreate(&usStop);

}

void us::free_arrays() {
	cudaFree(atomList_h);
	cudaFree(mass_h);
	cudaFree(groupComPos_h);
	fclose(cvFile);
}
void us::free_arrays_gpu() {
	cudaFree(atomList_d);
	cudaFree(mass_d);
	cudaFree(groupComPos_d);
}
