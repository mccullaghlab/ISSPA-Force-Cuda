
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "stringlib.h"

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
			} else if (strncmp(token,"inputCoord",10)==0) {
				strncpy(temp,strtok(NULL,search),MAXLEN);
				strcpy(inputCoordFileName,trim(temp));
				printf("input coordinate file name:%s\n",inputCoordFileName);
			}
		}
		fclose( inFile );
	}
	dt = 0.002*20.455;
	T = 298.0 * 0.00198717;
	pnu = 0.001f;
	nSteps = 100000;
	deltaWrite = 100;
	lbox = 200.0;
	deltaNN = 10;
	rcut = 12.0;
	rcut2 = rcut*rcut;
	rNN = 15.0;
	rNN2 = rNN*rNN;
	nMC = 10;


}

