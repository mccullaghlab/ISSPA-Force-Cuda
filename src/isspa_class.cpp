#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include "constants.h"
#include <iostream>

using namespace std;
#include "isspa_class.h"


void isspa::allocate(int nAtoms, int configMC)
{
	int i;
	nMC = configMC;
	//
	int nTypeBytes = nTypes*sizeof(float);
	// allocate atom based parameter arrays
	isspaTypes_h = (int *)malloc(nAtoms*sizeof(int));
	rmax_h = (float *)malloc(nTypeBytes);
	vtot_h = (float *)malloc(nTypeBytes);
	cudaMallocHost((float **) &isspaForceTable_h, nTypes*nRs*sizeof(float)); // force table
	cudaMallocHost((float **) &isspaForceR_h, nRs*sizeof(float)); // distance values for force table
	cudaMallocHost((float **) &isspaGTable_h, nTypes*nGRs*sizeof(float)); // force table
	cudaMallocHost((float **) &isspaGR_h, nGRs*sizeof(float)); // distance values for force table
	cudaMallocHost((float **) &isspaETable_h, nTypes*nERs*sizeof(float)); // force table
	cudaMallocHost((float **) &isspaER_h, nERs*sizeof(float)); // distance values for force table
}

void isspa::construct_parameter_arrays()
{
  int i;
  float x1;
  float x2;

  // compute other parameters needed for isspa force calculateions
  for (i=0;i<nTypes;i++) {
    vtot_h[i] = 4.0/3.0*PI*rmax_h[i]*rmax_h[i]*rmax_h[i]*0.0074/((float) nMC);// Monte Carlo integration normalization
  }  
}

void isspa::read_isspa_prmtop(char* isspaPrmtopFileName, int configMC)
{
	char line[MAXCHAR];
	char const *FlagSearch = "\%FLAG";
	char const *blank = " ";
	char const *metaDataFlag = "POINTERS";
	char const *isspaTypeFlag = "ISSPA_TYPE_INDEX";
	char const *isspaRmaxFlag = "ISSPA_RMAX";
	char const *isspaForcesFlag = "ISSPA_FORCES";
	char const *isspaDensitiesFlag = "ISSPA_DENSITIES";	
	char const *isspaEFieldFlag = "ISSPA_EFIELD";	
	char *flag;
	char *token;
	char *temp;
	int i, nLines;
	int atomCount;
	int typeCount;
	int lineCount;
	int nAtoms;

	FILE *prmFile = fopen(isspaPrmtopFileName, "r");

	if ( prmFile != NULL) {
		while (fgets(line, MAXCHAR, prmFile) != NULL) {
			if (strncmp(line,FlagSearch,5)==0) {
				token = strtok(line, blank);
				flag = strtok(NULL, blank);
				if (strncmp(flag,metaDataFlag,8) == 0) {
					// read meta data
					printf("Reading system metadata from ISSPA prmtop file\n");
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* read meta data section line by line */
					/* line 1: */
					temp = fgets(line, MAXCHAR, prmFile);
					nAtoms = atoi(strncpy(token,line,8));
					printf("Number of atoms from prmtop file: %d\n", nAtoms);
					nTypes = atoi(strncpy(token,line+8,8));
					printf("Number of ISSPA types from prmtop file: %d\n", nTypes);
					nRs = atoi(strncpy(token,line+32,8));
					printf("Number of ISSPA force values per type in prmtop file: %d\n", nRs);
					nGRs = atoi(strncpy(token,line+16,8));
					printf("Number of g force values per type in prmtop file: %d\n", nGRs);
					nERs = atoi(strncpy(token,line+24,8));
					printf("Number of electric field values per type in prmtop file: %d\n", nERs);
					allocate(nAtoms,configMC);
				} else if (strncmp(flag,isspaTypeFlag,16) == 0) {
					// 
					nLines = (int) (nAtoms + 9) / 10.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					atomCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (atomCount < nAtoms && lineCount < 10) {
							isspaTypes_h[atomCount] = atoi(strncpy(token,line+lineCount*8,8))-1;// minus one for C zero indexing
							atomCount++;
							lineCount++;
						}
					}
				} else if (strncmp(flag,isspaRmaxFlag,10) == 0) {
					// 
					nLines = (int) (nTypes + 4) / 5.0 ;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					typeCount = 0;
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						lineCount = 0;
						while (typeCount < nTypes && lineCount < 5) {
							rmax_h[typeCount] = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
				} else if (strncmp(flag,isspaDensitiesFlag,15) == 0) {
					//
				        nLines = nGRs;
				        /* skip format line */
				        temp = fgets(line, MAXCHAR, prmFile);
				        /* loop over lines */
					for (i=0;i<nLines;i++) {
					        temp = fgets(line, MAXCHAR, prmFile);
				                isspaGR_h[i] = atof(strncpy(token,line,16));
						for (typeCount=0;typeCount<nTypes;typeCount++) {
				                        isspaGTable_h[typeCount*nGRs+i] = atof(strncpy(token,line+(typeCount+1)*16,16));
						}
				        }
				        // store min and bin size
				        gRparams.x = isspaGR_h[0];  // min
				        gRparams.y = isspaGR_h[1] - isspaGR_h[0]; // bin size
				} else if (strncmp(flag,isspaEFieldFlag,12) == 0) {
				        //
				        nLines = nERs;
					/* skip format line */
				        temp = fgets(line, MAXCHAR, prmFile);
				        /* loop over lines */
					for (i=0;i<nLines;i++) {
				                temp = fgets(line, MAXCHAR, prmFile);
						isspaER_h[i] = atof(strncpy(token,line,16));
						for (typeCount=0;typeCount<nTypes;typeCount++) {
						  isspaETable_h[typeCount*nERs+i] = atof(strncpy(token,line+(typeCount+1)*16,16));
						}
				        }
				        // store min and bin size
				        eRparams.x = isspaER_h[0];  // min
				        eRparams.y = isspaER_h[1] - isspaER_h[0]; // bin size
				} else if (strncmp(flag,isspaForcesFlag,12) == 0) {
				       // 
				       nLines = nRs;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						isspaForceR_h[i] = atof(strncpy(token,line,16));
						for (typeCount=0;typeCount<nTypes;typeCount++) { 
							isspaForceTable_h[typeCount*nRs+i] = atof(strncpy(token,line+(typeCount+1)*16,16));
						}
					}
					// store min and bin size
					forceRparams.x = isspaForceR_h[0];
					forceRparams.y = isspaForceR_h[1] - isspaForceR_h[0];		 
				}
			}
		}
		fclose( prmFile );
	}
	// make other parameter arrays from the ones populated reading the prmtop
	construct_parameter_arrays();
}

void isspa::initialize_gpu(int nAtoms, int seed)
{
	int nTypeBytes = nTypes*sizeof(float);
	// allocate tabulated forces on device and pass data from host
	cudaMalloc((void **) &isspaForceTable_d, nTypes*nRs*sizeof(float));
	cudaMemcpy(isspaForceTable_d, isspaForceTable_h, nTypes*nRs*sizeof(float), cudaMemcpyHostToDevice);	
	// allocate tabulated densities on device and pass data from host
	cudaMalloc((void **) &isspaGTable_d, nTypes*nGRs*sizeof(float));
	cudaMemcpy(isspaGTable_d, isspaGTable_h, nTypes*nGRs*sizeof(float), cudaMemcpyHostToDevice);
	// allocate tabulated electric field on device and pass data from host
	cudaMalloc((void **) &isspaETable_d, nTypes*nERs*sizeof(float));
	cudaMemcpy(isspaETable_d, isspaETable_h, nTypes*nERs*sizeof(float), cudaMemcpyHostToDevice);
	// allocate MC position array on device
	cudaMalloc((void **) &mcpos_d, nAtoms*nMC*sizeof(float4));
	//cudaMemcpy(mcDist_d, mcDist_h, nTypes*sizeof(float4), cudaMemcpyHostToDevice);
	// allocate ISSPA types on device and pass data from host
	cudaMalloc((void **) &isspaTypes_d, nAtoms*sizeof(int));
	cudaMemcpy(isspaTypes_d, isspaTypes_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);
	// allocate rmax on device and pass data from host
	cudaMalloc((void **) &rmax_d, nTypeBytes);
	cudaMemcpy(rmax_d, rmax_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);	
	// allocate vtot on device and pass data from host
	cudaMalloc((void **) &vtot_d, nTypeBytes);
	cudaMemcpy(vtot_d, vtot_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);
	// allocate enow on device and pass data from host
	cudaMalloc((void **) &enow_d, nAtoms*nMC*sizeof(float4));
	// allocate e0now on device and pass data from host
	cudaMalloc((void **) &e0now_d, nAtoms*nMC*sizeof(float4));
	// random number states
	cudaMalloc((void**) &randStates_d, nAtoms*nMC*sizeof(curandState));
	init_rand_states(randStates_d, seed, nMC*nAtoms);
	// gpu timing
	cudaEventCreate(&isspaStart);
	cudaEventCreate(&isspaStop);

}
void isspa::free_arrays() {
	free(rmax_h); 
	free(vtot_h);
	cudaFree(isspaForceTable_h);	
	cudaFree(isspaForceR_h);
	cudaFree(isspaGTable_h);
	cudaFree(isspaGR_h);
	cudaFree(isspaETable_h);
	cudaFree(isspaER_h);
	//free(lj_h);
}
void isspa::free_arrays_gpu() {
	// free device variables
	cudaFree(isspaForceTable_d);
	cudaFree(isspaGTable_d);
	cudaFree(isspaETable_d);
	cudaFree(vtot_d);
	cudaFree(rmax_d); 
	cudaFree(randStates_d);
	cudaFree(isspaTypes_d);
	cudaFree(mcpos_d);
	cudaFree(enow_d);
	cudaFree(e0now_d);
	//cudaFree(lj_d);
}
