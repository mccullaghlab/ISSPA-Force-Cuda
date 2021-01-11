
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string.h>
#include "stringlib.h"
#include "constants_cuda.cuh"
using namespace std;
#include "config_class.h"


void config::initialize(char *cfgFileName)
{
	char *token;
	char *firstToken;
	char temp[MAXCHAR];
	char const *search = "=";
	char const *comment = "#";
	char line[MAXCHAR]; // maximum number of character per line set to 128
	FILE *inFile = fopen(cfgFileName,"r");

	printf("Reading config file: %s\n", cfgFileName);

	if ( inFile != NULL) {
		while (fgets(line, MAXCHAR, inFile) != NULL) {
			firstToken = strtok(line,comment);
			token = trim(strtok(firstToken, search));
			if (strncmp(token,"prmtop",6)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(prmtopFileName,trim(temp));
				printf("prmtop file name: %s\n",prmtopFileName);
			} else if (strncmp(token,"isspaPrmtop",11)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(isspaPrmtopFileName,trim(temp));
				printf("ISSPA prmtop file name: %s\n",isspaPrmtopFileName);
			} else if (strncmp(token,"posFile",7)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(posOutFileName,trim(temp));
				printf("Position trajectory written to file: %s\n",posOutFileName);
			} else if (strncmp(token,"posRstFile",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(posRstFileName,trim(temp));
				printf("Position restart written to file: %s\n",posRstFileName);
			} else if (strncmp(token,"velFile",7)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(velOutFileName,trim(temp));
				printf("Velocity trajectory written to file: %s\n",velOutFileName);
			} else if (strncmp(token,"velRstFile",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(velRstFileName,trim(temp));
				printf("Velocity restart written to file: %s\n",velRstFileName);
			} else if (strncmp(token,"forFile",7)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(forOutFileName,trim(temp));
				printf("Force trajectory written to file: %s\n",forOutFileName);
			} else if (strncmp(token,"inputCoord",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(inputCoordFileName,trim(temp));
				printf("input coordinate file name: %s\n",inputCoordFileName);
			} else if (strncmp(token,"nMC",3)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				nMC = atoi(trim(temp));
				printf("Number of Monte Carlo points per atom: %d\n",nMC);
			} else if (strncmp(token,"temperature",11)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				T = atof(trim(temp));
				printf("Temperature (K): %f\n",T);
			} else if (strncmp(token,"seed",4)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				seed = atof(trim(temp));
				printf("Random number seed: %i\n",seed);
			} else if (strncmp(token,"cutoff",6)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				rCut = atof(trim(temp));
				printf("Cutoff distance (Angstroms): %f\n",rCut);
			} else if (strncmp(token,"dielectric",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				eps = atof(trim(temp));
				printf("Dielectric: %f\n",eps);
			} else if (strncmp(token,"nSteps",6)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				nSteps = atoi(trim(temp));
				printf("Number of MD steps: %d\n",nSteps);
			} else if (strncmp(token,"deltaWrite",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				deltaWrite = atoi(trim(temp));
				printf("Write frequency: %d\n",deltaWrite);
			} else if (strncmp(token,"boxLength",9)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				lbox = atof(trim(temp));
				printf("Box Length (Angstroms): %f\n",lbox);
				printf("Box Volume (Angstroms^3): %f\n",lbox*lbox*lbox);
			} else if (strncmp(token,"pnu",3)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				pnu = atof(trim(temp));
				printf("pnu (Anderson Thermostat Frequency): %f\n",pnu);
			} else if (strncmp(token,"velRst",6)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				velRst = atoi(trim(temp));
				if (velRst == 1) {
					printf("Restarting from velocities read from file.\n");
				}
			} else if (strncmp(token,"inputVel",8)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(inputVelFileName,trim(temp));
				printf("Input velocity file name: %s\n",inputVelFileName);
			} else if (strncmp(token,"US",2)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				us = atoi(trim(temp));
				if (us == 1) {
					printf("Harmonic bias sampling is on.\n");
				}
			} else if (strncmp(token,"usCfgFile",9)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(usCfgFileName,trim(temp));
				printf("Harmonic bias file name: %s\n",usCfgFileName);
			}
		}
		fclose( inFile );
	}
	dtPs = 0.002;  // currently the time step is hard coded
	dt = dtPs*20.455; // convert to amber time units
	T *= 0.00198717; // convert to energy units
	rCut2 = rCut*rCut; // cutoff distance squared
	//pnu = 0.03f;  // Anderson thermostat frequency
	deltaNN = 10;  // neighborlist step frequency - currently not used
	rNN = 15.0;    // neighborlist distance - currently not used
	rNN2 = rNN*rNN; // neighbor list distance squared - currently not used
	//seed = 12345;  // random number seed currently hard coded to be changed later
	set_cuda_constants();
}

void config::set_cuda_constants()
{
	cudaMemcpyToSymbol(&dt_d, &dt, sizeof(float),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&pnu_d, &pnu, sizeof(float),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&T_d, &T, sizeof(float),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&lbox_d, &lbox, sizeof(float),0, cudaMemcpyHostToDevice);
}

