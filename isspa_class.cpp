


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


void isspa::allocate(int nAtoms)
{
	int i;
	// temp hard code these values
	nTypes = 1;
	nMC = 10;
	//
	int nTypeBytes = nTypes*sizeof(float);
	// allocate atom based parameter arrays
	isspaTypes_h = (int *)malloc(nAtoms*sizeof(int));
	lj_h = (float2 *)malloc(nTypes*sizeof(float2));
	x0_h = (float *)malloc(nTypeBytes);
	g0_h = (float *)malloc(nTypeBytes);
	gr2_h = (float2 *)malloc(nTypes*sizeof(float2));
	w_h = (float *)malloc(nTypeBytes);
	alpha_h = (float *)malloc(nTypeBytes);
	vtot_h = (float *)malloc(nTypeBytes);
	lj_vtot_h = (float4 *)malloc(nTypes*sizeof(float4));;     // isspa LJ parameter
	x0_w_h = (float2 *)malloc(nTypes*sizeof(float2));;     // x0 and w parameters
	gr2_g0_alpha_h = (float4 *)malloc(nTypes*sizeof(float4));;     // gr2 g0 alpha
	// temp one type
	for (i=0;i<nAtoms;i++) {
		isspaTypes_h[i] = 0;
	}
	gr2_h[0].x = 11.002;
	gr2_h[0].y = 21.478;
	w_h[0] = 0.801;
	g0_h[0] = 1.714; // height of parabola
	x0_h[0] = 4.118;
	alpha_h[0] = 2.674; 
	vtot_h[0] = 16.0/3.0*3.1415926535*w_h[0]*g0_h[0]/((float) nMC)*0.0334*1E-2;
	lj_h[0].x = 6.669e7;
	lj_h[0].y = 1.103e4;

	lj_vtot_h[0].x = lj_h[0].x;
	lj_vtot_h[0].y = lj_h[0].y;
	lj_vtot_h[0].z = vtot_h[0];
	x0_w_h[0].x = x0_h[0];
	x0_w_h[0].y = w_h[0];
	gr2_g0_alpha_h[0].x = gr2_h[0].x;
	gr2_g0_alpha_h[0].y = gr2_h[0].y;
	gr2_g0_alpha_h[0].z = g0_h[0];
	gr2_g0_alpha_h[0].w = alpha_h[0];

}

void isspa::initialize_gpu(int nAtoms, int seed)
{
	int nTypeBytes = nTypes*sizeof(float);
	// allocate atom based parameter arrays
	cudaMalloc((void **) &mcpos_d, nAtoms*nMC*sizeof(float4));
	cudaMalloc((void **) &lj_vtot_d, nTypes*sizeof(float4));
	cudaMalloc((void **) &x0_w_d, nTypes*sizeof(float2));
	cudaMalloc((void **) &gr2_g0_alpha_d, nTypes*sizeof(float4));
	//cudaMalloc((void **) &lj_d, nTypes*sizeof(float2));
	cudaMalloc((void **) &isspaTypes_d, nAtoms*sizeof(int));
	cudaMalloc((void **) &x0_d, nTypeBytes);
	//cudaMalloc((void **) &g0_d, nTypeBytes);
	//cudaMalloc((void **) &gr2_d, nTypes*sizeof(float2));
	//cudaMalloc((void **) &w_d, nTypeBytes);
	//cudaMalloc((void **) &alpha_d, nTypeBytes);
	//cudaMalloc((void **) &vtot_d, nTypeBytes);
	// copy params to gpu
	cudaMemcpy(isspaTypes_d, isspaTypes_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(lj_vtot_d, lj_vtot_h, nTypes*sizeof(float4), cudaMemcpyHostToDevice);	
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
	free(lj_h);
}
void isspa::free_arrays_gpu() {
	// free device variables
	cudaFree(w_d); 
	cudaFree(g0_d); 
	cudaFree(gr2_d); 
	cudaFree(x0_d); 
	cudaFree(alpha_d); 
	cudaFree(vtot_d); 
	cudaFree(randStates_d);
	cudaFree(isspaTypes_d);
	cudaFree(mcpos_d);
	cudaFree(lj_d);
}
