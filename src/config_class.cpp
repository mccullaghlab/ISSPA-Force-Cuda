
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
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
			} else if (strncmp(token,"inputCoord",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(inputCoordFileName,trim(temp));
				printf("input coordinate file name: %s\n",inputCoordFileName);
			} else if (strncmp(token,"nMC",3)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				nMC = atoi(trim(temp));
				printf("Number of Monte Carlo points: %d\n",nMC);
			} else if (strncmp(token,"nSteps",6)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				nSteps = atoi(trim(temp));
				printf("Number of MD steps: %d\n",nSteps);
			} else if (strncmp(token,"deltaWrite",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				deltaWrite = atoi(trim(temp));
				printf("Write frequency: %d\n",deltaWrite);
			}
		}
		fclose( inFile );
	}
	dtPs = 0.002;
	dt = dtPs*20.455; // convert to amber time units
	T = 298.0 * 0.00198717; // convert to energy units
	pnu = 0.001f;
	lbox = 200.0;
	deltaNN = 10;
	rCut = 12.0;
	rCut2 = rCut*rCut;
	rNN = 15.0;
	rNN2 = rNN*rNN;
	seed = 12345;
	set_cuda_constants();

}

void config::set_cuda_constants()
{
	cudaMemcpyToSymbol(&dt_d, &dt, sizeof(float),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&pnu_d, &pnu, sizeof(float),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&T_d, &T, sizeof(float),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&lbox_d, &lbox, sizeof(float),0, cudaMemcpyHostToDevice);
}

