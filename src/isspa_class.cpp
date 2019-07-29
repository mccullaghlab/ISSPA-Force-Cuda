


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include "constants.h"

using namespace std;
#include "isspa_class.h"


void isspa::allocate(int nAtoms, int configMC)
{
	int i;
	nMC = configMC;
	// allocate atom based parameter arrays
	isspaTypes_h = (int *)malloc(nAtoms*sizeof(int));
	// tab force arrays
	cudaMallocHost((float **) &isspaForceTable_h, nTypes*nForceRs*sizeof(float)); // force table
	cudaMallocHost((float **) &isspaForceR_h, nForceRs*sizeof(float)); // distance values for force table
	// tab G arrays
	cudaMallocHost((float **) &isspaGTable_h, nTypes*nGRs*sizeof(float)); // force table
	cudaMallocHost((float **) &isspaGR_h, nGRs*sizeof(float)); // distance values for force table
	// MC distribution arrays
	cudaMallocHost((float4 **) &mcDist_h, nTypes*sizeof(float4));

}

void isspa::construct_parameter_arrays()
{
	int i;
	// finish MC dist parameters
	for (i=0;i<nTypes;i++) {
		mcDist_h[i].z = mcDist_h[i].y - mcDist_h[i].x; // domain size
		mcDist_h[i].w = 4.0 * PI * mcDist_h[i].z/float(nMC)*RHO;      // normalization factor
		printf("%10.5f%10.5f%10.5f%10.5f\n", mcDist_h[i].x, mcDist_h[i].y, mcDist_h[i].z,mcDist_h[i].w);	
	}

}

void isspa::read_isspa_prmtop(char* isspaPrmtopFileName, int configMC)
{
	char line[MAXCHAR];
	char const *FlagSearch = "\%FLAG";
	char const *blank = " ";
	char const *metaDataFlag = "POINTERS";
	char const *isspaTypeFlag = "ISSPA_TYPE_INDEX";
	char const *isspaMCMinFlag = "ISSPA_MCMIN";
	char const *isspaMCMaxFlag = "ISSPA_MCMAX";
	char const *isspaForcesFlag = "ISSPA_FORCES";
	char const *isspaDensitiesFlag = "ISSPA_DENSITIES";
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
					nGRs = atoi(strncpy(token,line+16,8));
					printf("Number of ISSPA g values per type in prmtop file: %d\n", nGRs);
					nForceRs = atoi(strncpy(token,line+24,8));
					printf("Number of ISSPA force values per type in prmtop file: %d\n", nForceRs);
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
				} else if (strncmp(flag,isspaMCMinFlag,11) == 0) {
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
							mcDist_h[typeCount].x = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
				} else if (strncmp(flag,isspaMCMaxFlag,11) == 0) {
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
							mcDist_h[typeCount].y = atof(strncpy(token,line+lineCount*16,16));
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
				} else if (strncmp(flag,isspaForcesFlag,12) == 0) {
					// 
					nLines = nForceRs;
					/* skip format line */
					temp = fgets(line, MAXCHAR, prmFile);
					/* loop over lines */
					for (i=0;i<nLines;i++) {
						temp = fgets(line, MAXCHAR, prmFile);
						isspaForceR_h[i] = atof(strncpy(token,line,16));
						for (typeCount=0;typeCount<nTypes;typeCount++) { 
							isspaForceTable_h[typeCount*nForceRs+i] = atof(strncpy(token,line+(typeCount+1)*16,16));
						}
					}
					// store min and bin size
					forceRparams.x = isspaForceR_h[0];  // min
					forceRparams.y = isspaForceR_h[1] - isspaForceR_h[0]; // bin size
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
	cudaMalloc((void **) &isspaForceTable_d, nTypes*nForceRs*sizeof(float));
	cudaMemcpy(isspaForceTable_d, isspaForceTable_h, nTypes*nForceRs*sizeof(float), cudaMemcpyHostToDevice);	
	// allocate tabulated densities on device and pass data from host
	cudaMalloc((void **) &isspaGTable_d, nTypes*nGRs*sizeof(float));
	cudaMemcpy(isspaGTable_d, isspaGTable_h, nTypes*nGRs*sizeof(float), cudaMemcpyHostToDevice);
	// allocate MC distribution parameters on device and pass data from host
	cudaMalloc((void **) &mcDist_d, nTypes*sizeof(float4));	
	cudaMemcpy(mcDist_d, mcDist_h, nTypes*sizeof(float4), cudaMemcpyHostToDevice);	
	// allocate MC position array on device
	cudaMalloc((void **) &mcPos_d, nAtoms*nMC*sizeof(float4));
	// allocate ISSPA types on device and pass data from host
	cudaMalloc((void **) &isspaTypes_d, nAtoms*sizeof(int));
	cudaMemcpy(isspaTypes_d, isspaTypes_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);	
	// random number states
	cudaMalloc((void**) &randStates_d, nAtoms*nMC*sizeof(curandState));
	init_rand_states(randStates_d, seed, nMC*nAtoms);
	// intialize gpu timing events
	cudaEventCreate(&isspaStart);
	cudaEventCreate(&isspaStop);

}
void isspa::free_arrays() {
	cudaFree(isspaForceTable_h);	
	cudaFree(isspaForceR_h);
	cudaFree(isspaGTable_h);	
	cudaFree(isspaGR_h);
	cudaFree(mcDist_h);
	free(isspaTypes_h);
}
void isspa::free_arrays_gpu() {
	// free device variables
	cudaFree(isspaForceTable_d);
	cudaFree(isspaGTable_d);
	cudaFree(mcDist_d);
	cudaFree(randStates_d);
	cudaFree(isspaTypes_d);
	cudaFree(mcPos_d);
}
