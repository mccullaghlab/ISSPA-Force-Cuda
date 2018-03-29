#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define nDim 3

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

__global__ void init_rand(unsigned int long seed, curandState_t* states){
	curand_init(seed,blockIdx.x,0,&states);
}

__global__ void parab_sphere(float *xyz, float w, float x0, unsigned int N) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float rnow;
	float prob;
	float attempt;
	float x1, x2, r2;
	curandState_t state;

	if (index < N)
	{
		curand_init(0,blockIdx.x,0,&state);
		rnow = 1.0f - 2.0f * curand_uniform(&state);
		prob = rnow*rnow;
		attempt = curand_uniform(&state);
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * curand_uniform(&state);
			prob = rnow*rnow;
			attempt = curand_uniform(&state);
		}
		rnow = w * rnow + x0;
		x1 = 1.0f - 2.0f * curand_uniform(&state);
		x2 = 1.0f - 2.0f * curand_uniform(&state);
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f) 
		{
			x1 = 1.0f - 2.0f * curand_uniform(&state);
                	x2 = 1.0f - 2.0f * curand_uniform(&state);
			r2 = x1*x1 + x2*x2;
		}
		xyz[index*nDim] = rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		xyz[index*nDim+1] = rnow*x1*r2;
		xyz[index*nDim+2] = rnow*x2*r2;

	}
}


int main(void) 
{
	float *xyz_h; // host data
	float *xyz_d; // device data
	float x0=2.0;
	float w=1.0;
	int long N = 1e4;
	int nBytes, i;
	cudaEvent_t start, stop;
	float milliseconds;
	unsigned int long seed = 12345;
	curandState_t *states;

	// start device timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// size of xyz arrays
	nBytes = N*nDim*sizeof(float);
 	// allocate local variables
	xyz_h = (float *)malloc(nBytes);
	// allocate device variables
	cudaMalloc((void **) &xyz_d, nBytes);
	cudaMalloc((void **) &states, N*sizeof(curandState_t));

	// run parabola random cuda kernal
	init_rand<<<N,1>>>(seed,states);
	parab_sphere<<<N,1>>>(xyz_d, w, x0, N, states);

	// pass device variable, a_d, to host variable a_h
	cudaMemcpy(xyz_h, xyz_d, nBytes, cudaMemcpyDeviceToHost);	

	// get GPU time
	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
//printf("Time = %12.8f\n", milliseconds);

	// print xyz file
	xyzFile = fopen("sphere.xyz","w");
	fprintf(xyzFile,"%ld\n", N);
	fprintf(xyzFile,"%ld\n", N);
	for (i=0;i<N; i++) 
	{
		fprintf(xyzFile,"C %10.6f %10.6f %10.6f\n", xyz_h[i*nDim],xyz_h[i*nDim+1],xyz_h[i*nDim+2]);
	}
	fclose(xyzFile)
	// free host variables
	free(xyz_h);
	// free device variables
	cudaFree(xyz_d); 
	cudaFree(states);

	return 0;

}


