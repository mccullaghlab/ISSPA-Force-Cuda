


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
	//
	int nTypeBytes = nTypes*sizeof(float);
	// allocate atom based parameter arrays
	isspaTypes_h = (int *)malloc(nAtoms*sizeof(int));
	//lj_h = (float2 *)malloc(nTypes*sizeof(float2));
	x0_h = (float *)malloc(nTypeBytes);
	g0_h = (float *)malloc(nTypeBytes);
	gr2_h = (float2 *)malloc(nTypes*sizeof(float2));
	w_h = (float *)malloc(nTypeBytes);
	alpha_h = (float *)malloc(nTypeBytes);
	vtot_h = (float *)malloc(nTypeBytes);
	cudaMallocHost((float **) &isspaForceTable_h, nTypes*nRs*sizeof(float)); // force table
	cudaMallocHost((float **) &isspaForceR_h, nRs*sizeof(float)); // distance values for force table
	// combined parameter data
	//lj_vtot_h = (float4 *)malloc(nTypes*sizeof(float4));;     // isspa LJ parameter
	x0_w_h = (float2 *)malloc(nTypes*sizeof(float2));;     // x0 and w parameters
	gr2_g0_alpha_h = (float4 *)malloc(nTypes*sizeof(float4));;     // gr2 g0 alpha

}

void isspa::construct_parameter_arrays()
{
	int i;
	float x1;
	float x2;
	// compute other version of parabola parameters from given g0, x0 and alpha
	for (i=0;i<nTypes;i++) {
		x1 = x0_h[i] - sqrt(g0_h[i]/alpha_h[i]);	 			// lower limit of parabola
		x2 = x0_h[i] + sqrt((g0_h[i]-1.0)/alpha_h[i]);	 			// upper limit of parabola
		printf("%8.3f %8.3f\n", x1, x2);
		gr2_h[i].x = x1*x1;							// square of lower limit of parabola (g(r) = 0)
		gr2_h[i].y = x2*x2;							// square of upper limit of parabola (g(r) = 1)
		w_h[i] = x2-x1;	 							// width of parabola
		vtot_h[i] = 16.0/3.0*PI*w_h[i]*g0_h[i]/((float) nMC)*0.0334*1E-2;	// Monte Carlo integration normalization
		// store these values in float2 and float4s for computational efficiency
		x0_w_h[i].x = x0_h[i];
		x0_w_h[i].y = w_h[i];
		gr2_g0_alpha_h[i].x = gr2_h[i].x;
		gr2_g0_alpha_h[i].y = gr2_h[i].y;
		gr2_g0_alpha_h[i].z = g0_h[i];
		gr2_g0_alpha_h[i].w = alpha_h[i];
	}

}

void isspa::read_isspa_prmtop(char* isspaPrmtopFileName, int configMC)
{
	char line[MAXCHAR];
	char const *FlagSearch = "\%FLAG";
	char const *blank = " ";
	char const *metaDataFlag = "POINTERS";
	char const *isspaTypeFlag = "ISSPA_TYPE_INDEX";
	char const *isspaG0Flag = "ISSPA_G0";
	char const *isspaX0Flag = "ISSPA_X0";
	char const *isspaAlphaFlag = "ISSPA_ALPHA";
	char const *isspaForcesFlag = "ISSPA_FORCES";
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
					nRs = atoi(strncpy(token,line+16,8));
					printf("Number of ISSPA force values per type in prmtop file: %d\n", nRs);
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
				} else if (strncmp(flag,isspaG0Flag,8) == 0) {
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
							g0_h[typeCount] = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
				} else if (strncmp(flag,isspaX0Flag,8) == 0) {
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
							x0_h[typeCount] = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
				} else if (strncmp(flag,isspaAlphaFlag,11) == 0) {
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
							alpha_h[typeCount] = atof(strncpy(token,line+lineCount*16,16));
							typeCount++;
							lineCount++;
						}
					}
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
					printf("%8.3f %8.3f\n", forceRparams.x, forceRparams.y);
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
	// allocate atom based parameter arrays
	cudaMalloc((void **) &isspaForceTable_d, nTypes*nRs*sizeof(float));
	cudaMemcpy(isspaForceTable_d, isspaForceTable_h, nTypes*nRs*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMalloc((void **) &mcpos_d, nAtoms*nMC*sizeof(float4));
	cudaMalloc((void **) &x0_w_d, nTypes*sizeof(float2));
	cudaMalloc((void **) &gr2_g0_alpha_d, nTypes*sizeof(float4));
	//cudaMalloc((void **) &lj_d, nTypes*sizeof(float2));
	cudaMalloc((void **) &isspaTypes_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &x0_d, nTypeBytes);
	//cudaMalloc((void **) &g0_d, nTypeBytes);
	//cudaMalloc((void **) &gr2_d, nTypes*sizeof(float2));
	//cudaMalloc((void **) &w_d, nTypeBytes);
	//cudaMalloc((void **) &alpha_d, nTypeBytes);
	cudaMalloc((void **) &vtot_d, nTypeBytes);
	// copy params to gpu
	cudaMemcpy(isspaTypes_d, isspaTypes_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);	
	//cudaMemcpy(lj_vtot_d, lj_vtot_h, nTypes*sizeof(float4), cudaMemcpyHostToDevice);	
	cudaMemcpy(vtot_d, vtot_h, nTypes*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(x0_w_d, x0_w_h, nTypes*sizeof(float2), cudaMemcpyHostToDevice);	
	cudaMemcpy(gr2_g0_alpha_d, gr2_g0_alpha_h, nTypes*sizeof(float4), cudaMemcpyHostToDevice);	
	//cudaMemcpy(lj_d, lj_h, nTypes*sizeof(float2), cudaMemcpyHostToDevice);	
	//cudaMemcpy(w_d, w_h, nTypeBytes, cudaMemcpyHostToDevice);	
	cudaMemcpy(x0_d, x0_h, nTypeBytes, cudaMemcpyHostToDevice);	
	//cudaMemcpy(g0_d, g0_h, nTypeBytes, cudaMemcpyHostToDevice);	
	//cudaMemcpy(gr2_d, gr2_h, nTypes*sizeof(float2), cudaMemcpyHostToDevice);	
	//cudaMemcpy(alpha_d, alpha_h, nTypeBytes, cudaMemcpyHostToDevice);	
	//cudaMemcpy(vtot_d, vtot_h, nTypeBytes, cudaMemcpyHostToDevice);	
	// random number states
	cudaMalloc((void**) &randStates_d, nAtoms*nMC*sizeof(curandState));
	init_rand_states(randStates_d, seed, nMC*nAtoms);
	// gpu timing
	cudaEventCreate(&isspaStart);
	cudaEventCreate(&isspaStop);

}
void isspa::free_arrays() {
	free(w_h); 
	free(g0_h); 
	free(gr2_h); 
	free(x0_h); 
	free(alpha_h); 
	free(vtot_h);
        free(x0_w_h);
	free(gr2_g0_alpha_h);
	cudaFree(isspaForceTable_h);	
	cudaFree(isspaForceR_h);
	//free(lj_h);
}
void isspa::free_arrays_gpu() {
	// free device variables
	//cudaFree(w_d); 
	//cudaFree(g0_d); 
	//cudaFree(gr2_d); 
	cudaFree(x0_d); 
	//cudaFree(alpha_d); 
	cudaFree(isspaForceTable_d);
	cudaFree(x0_w_d);
	cudaFree(gr2_g0_alpha_d);
	cudaFree(vtot_d); 
	cudaFree(randStates_d);
	cudaFree(isspaTypes_d);
	cudaFree(mcpos_d);
	//cudaFree(lj_d);
}
