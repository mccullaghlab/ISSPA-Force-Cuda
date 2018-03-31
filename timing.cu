#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define nDim 3
#define MC 10
//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

//__global__ void init_rand(unsigned int long seed, curandState_t* states){
//	curand_init(seed,blockIdx.x,0,&states);
//}

__global__ void ispa_force(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	float rnow;
	float prob;
	float attempt;
	float mc_pos[3];
	float mc_pos_atom[3];
	float x1, x2, r2;
	int atom;
	int atom2;
	int it;    // atom type of atom of interest
	int jt;    // atom type of other atom
	float gnow;
	float temp, dist2;
	int ev_flag, k;
	float rinv, r6, fs;
	curandState_t state;

	if (index < nAtoms*nMC)
	{
		// get atom number of interest
		atom = index%nAtoms;
		it = ityp[atom];
		// initialize random number generator
		curand_init(0,blockIdx.x,index,&state);
		// select one point from 1D parabolic distribution
		rnow = 1.0f - 2.0f * curand_uniform(&state);
		prob = rnow*rnow;
		attempt = curand_uniform(&state);
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * curand_uniform(&state);
			prob = rnow*rnow;
			attempt = curand_uniform(&state);
		}
		rnow = w[it] * rnow + x0[it];
		// select two points on surface of sphere
		x1 = 1.0f - 2.0f * curand_uniform(&state);
		x2 = 1.0f - 2.0f * curand_uniform(&state);
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f) 
		{
			x1 = 1.0f - 2.0f * curand_uniform(&state);
                	x2 = 1.0f - 2.0f * curand_uniform(&state);
			r2 = x1*x1 + x2*x2;
		}
		// generate 3D MC pos based on position on surface of sphere and parabolic distribution in depth
		mc_pos[0] = rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mc_pos[1] = rnow*x1*r2;
		mc_pos[2] = rnow*x2*r2;

		mc_pos_atom[0] = mc_pos[0] + xyz[atom*nDim];
		mc_pos_atom[1] = mc_pos[1] + xyz[atom*nDim+1];
		mc_pos_atom[2] = mc_pos[2] + xyz[atom*nDim+2];
		// compute density at MC point due to all other atoms
		gnow = 1.0f;
		ev_flag = 0;
		for (atom2=0;atom2<nAtoms;atom2++) 
		{
			if (atom2 != atom) 
			{
				jt = ityp[atom2];
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) 
				{
					temp = mc_pos_atom[k] - xyz[atom2*nDim+k];
					dist2 += temp*temp;
				}
				if (dist2 < gr2[jt*2]) {
					ev_flag = 1;	
					break;
				} else if (dist2 < gr2[jt*2+1]) {
					temp = sqrtf(dist2)-x0[jt];
					gnow *= (-alpha[jt] * temp*temp + g0[jt]);
				}
			}
		}
		
		if (ev_flag ==0) {
			rinv = 1.0f / rnow;
			r2 = rinv * rinv;
			r6 = r2 * r2 * r2;
			fs = gnow * r6 * (lj_B[it] - lj_A[it] * r6);
			atomicAdd(&f[atom*nDim], fs*mc_pos[0]);
			atomicAdd(&f[atom*nDim+1], fs*mc_pos[1]);
			atomicAdd(&f[atom*nDim+2], fs*mc_pos[2]);
		}

	}
}


void cpu_ispa_force(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC) {
	float rnow;
	float prob;
	float attempt;
	float mc_pos[3];
	float mc_pos_atom[3];
	float x1, x2, r2;
	int atom;
	int atom2;
	int it;    // atom type of atom of interest
	int jt;    // atom type of other atom
	float gnow;
	float temp, dist2;
	int ev_flag, k, index;
	float rinv, r6, fs;

	for (index=0;index<nAtoms*nMC;index++) 
	{
		// get atom number of interest
		atom = index%nAtoms;
		it = ityp[atom];
		// initialize random number generator
//		curand_init(0,blockIdx.x,index,&state);
		// select one point from 1D parabolic distribution
		rnow = 1.0f - 2.0f * (float) rand() / (float) RAND_MAX; 
		prob = rnow*rnow;
		attempt = (float) rand() / (float) RAND_MAX;
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * (float) rand() / (float) RAND_MAX;
			prob = rnow*rnow;
			attempt = (float) rand() / (float) RAND_MAX;
		}
		rnow = w[it] * rnow + x0[it];
		// select two points on surface of sphere
		x1 = 1.0f - 2.0f * (float) rand() / (float) RAND_MAX;
		x2 = 1.0f - 2.0f * (float) rand() / (float) RAND_MAX;
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f) 
		{
			x1 = 1.0f - 2.0f * (float) rand() / (float) RAND_MAX;
                	x2 = 1.0f - 2.0f * (float) rand() / (float) RAND_MAX;
			r2 = x1*x1 + x2*x2;
		}
		// generate 3D MC pos based on position on surface of sphere and parabolic distribution in depth
		mc_pos[0] = rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
		mc_pos[1] = rnow*x1*r2;
		mc_pos[2] = rnow*x2*r2;

		mc_pos_atom[0] = mc_pos[0] + xyz[atom*nDim];
		mc_pos_atom[1] = mc_pos[1] + xyz[atom*nDim+1];
		mc_pos_atom[2] = mc_pos[2] + xyz[atom*nDim+2];
		// compute density at MC point due to all other atoms
		gnow = 1.0f;
		ev_flag = 0;
		for (atom2=0;atom2<nAtoms;atom2++) 
		{
			if (atom2 != atom) 
			{
				jt = ityp[atom2];
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) 
				{
					temp = mc_pos_atom[k] - xyz[atom2*nDim+k];
					dist2 += temp*temp;
				}
				if (dist2 < gr2[jt*2]) {
					ev_flag = 1;	
					break;
				} else if (dist2 < gr2[jt*2+1]) {
					temp = sqrtf(dist2)-x0[jt];
					gnow *= (-alpha[jt] * temp*temp + g0[jt]);
				}
			}
		}
		
		if (ev_flag ==0) {
			rinv = 1.0f / rnow;
			r2 = rinv * rinv;
			r6 = r2 * r2 * r2;
			fs = gnow * r6 * (lj_B[it] - lj_A[it] * r6);
			f[atom*nDim] += fs*mc_pos[0];
			f[atom*nDim+1] += fs*mc_pos[1];
			f[atom*nDim+2] += fs*mc_pos[2];
		}

	}
}
int main(void)  
{
	FILE *timeFile;
	int nMaxAtoms = 1000;
	int nAtoms;
	int nTrials = 10;
	float cpuTime[nTrials];
	float gpuTime[nTrials];
	int trial;
	int nAtomTypes = 1;
	int blockSize;      // The launch configurator returned block size 
    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    	int gridSize;       // The actual grid size needed, based on input size 
//	float *mc_xyz_h;    // MC coordinate array - host data
//	float *mc_xyz_d;    // MC coordinate array - device data
	float *xyz_h;    // coordinate array - host data
	float *xyz_d;    // coordinate array - device data
	float *f_h;      // force array - host data
	float *f_d;      // force array - device data
	int *ityp_h;     // atom type array - host data
	int *ityp_d;     // atom type array - device data
	float *x0_h;     // center position of parabola and g - host data 
	float *x0_d;     // center position of parabola and g - device data
	float *g0_h;     // height of parabola approximation of g - host data 
	float *g0_d;     // height of parabola approximation of g - device data
	float *gr2_h;     // excluded volume distance and end of parabola distance squared - host data 
	float *gr2_d;     // excluded volume distance and end of parabola distance squared - device data
	float *w_h;      // width of parabola - host data
	float *w_d;      // width of parabola - device data
	float *alpha_h;  // alpha parameter for g - host data
	float *alpha_d;  // alpha parameter for g - device data
	float *lj_A_h;   // Lennard-Jones A parameter - host data
	float *lj_A_d;   // Lennard-Jones A parameter - device data
	float *lj_B_h;   // Lennard-Jones B parameter - host data
	float *lj_B_d;   // Lennard-Jones B parameter - device data
	int nMC = MC;    // number of MC points
	int nAtomBytes, nTypeBytes, i;
	cudaEvent_t start, stop;
	float milliseconds;
	float avgGpuTime,avgCpuTime;
	float cpuStdev, gpuStdev, temp;
	unsigned int long seed = 12345;

	timeFile = fopen("timing.dat", "w");	

	for (nAtoms=10;nAtoms<nMaxAtoms;nAtoms += 10){
		fprintf(timeFile,"%10d",nAtoms);
		// size of xyz arrays
		nAtomBytes = nAtoms*sizeof(float);
		nTypeBytes = nAtomTypes*sizeof(float);
		// allocate atom coordinate arrays
		xyz_h = (float *)malloc(nAtomBytes*nDim);
		cudaMalloc((void **) &xyz_d, nAtomBytes*nDim);
		// allocate atom force arrays
		f_h = (float *)malloc(nAtomBytes*nDim);
		cudaMalloc((void **) &f_d, nAtomBytes*nDim);
		// allocate atom type arrays
		ityp_h = (int *)malloc(nAtoms*sizeof(int));
		cudaMalloc((void **) &ityp_d, nAtoms*sizeof(int));
		// allocate atom based parameter arrays
		x0_h = (float *)malloc(nTypeBytes);
		cudaMalloc((void **) &x0_d, nTypeBytes);
		g0_h = (float *)malloc(nTypeBytes);
		cudaMalloc((void **) &g0_d, nTypeBytes);
		gr2_h = (float *)malloc(nTypeBytes*2);
		cudaMalloc((void **) &gr2_d, nTypeBytes*2);
		w_h = (float *)malloc(nTypeBytes);
		cudaMalloc((void **) &w_d, nTypeBytes);
		alpha_h = (float *)malloc(nTypeBytes);
		cudaMalloc((void **) &alpha_d, nTypeBytes);
		lj_A_h = (float *)malloc(nTypeBytes);
		cudaMalloc((void **) &lj_A_d, nTypeBytes);
		lj_B_h = (float *)malloc(nTypeBytes);
		cudaMalloc((void **) &lj_B_d, nTypeBytes);
		
		
		// populate host arrays
		for (i=0;i<nAtoms;i++) {
			xyz_h[i*nDim] = (float) i*7.0;
			xyz_h[i*nDim+1] = xyz_h[i*nDim+2] = 0.0f;
			f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
			ityp_h[i] = 0;
		}
		gr2_h[0] = 11.002;
		gr2_h[1] = 21.478;
		w_h[0] = 0.801;
		g0_h[0] = 1.714; // height of parabola
		x0_h[0] = 4.118;
		alpha_h[0] = 2.674; 
		lj_A_h[0] = 6.669e7;
		lj_B_h[0] = 1.103e4;

		avgGpuTime = avgCpuTime = 0.0;
		for (trial=0;trial<nTrials;trial++) {		
			for (i=0;i<nAtoms;i++) {
				f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
			}
			// start device timer
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
			// copy data to device
			cudaMemcpy(f_d, f_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
			cudaMemcpy(xyz_d, xyz_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
			cudaMemcpy(ityp_d, ityp_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);	
			cudaMemcpy(w_d, w_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(x0_d, x0_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(g0_d, g0_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(gr2_d, gr2_h, 2*nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(alpha_d, alpha_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(lj_A_d, lj_A_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(lj_B_d, lj_B_h, nTypeBytes, cudaMemcpyHostToDevice);	
			
			// determine gridSize and blockSize
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ispa_force, 0, nAtoms*nMC); 
			
			// Round up according to array size 
			gridSize = (nAtoms*nMC + blockSize - 1) / blockSize; 
			
			//printf("gridSize = %d, blockSize = %d\n", gridSize, blockSize);
			// run parabola random cuda kernal
			ispa_force<<<gridSize, blockSize>>>(xyz_d, f_d, w_d, x0_d, g0_d, gr2_d, alpha_d, lj_A_d, lj_B_d, ityp_d, nAtoms, nMC);
			
			// pass device variable, f_d, to host variable f_h
			cudaMemcpy(f_h, f_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
			
			// get GPU time
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			gpuTime[trial] = milliseconds;
			avgGpuTime += milliseconds;
			// rezero forces
			for (i=0;i<nAtoms;i++) {
				f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
			}

			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
			cpu_ispa_force(xyz_h, f_h, w_h, x0_h, g0_h, gr2_h, alpha_h, lj_A_h, lj_B_h, ityp_h, nAtoms, nMC);
			// get CPU time
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			cpuTime[trial] = milliseconds;
			avgCpuTime += milliseconds;

//			fprintf(timeFile,"%12.8f", milliseconds);
		}
		temp = 0.0;
		avgCpuTime /= (float) nTrials;
		avgGpuTime /= (float) nTrials;
		cpuStdev = 0.0;
		gpuStdev = 0.0;
		for (trial=0;trial<nTrials;trial++) {
			temp = cpuTime[trial] - avgCpuTime;
			cpuStdev += temp*temp;
			temp = gpuTime[trial] - avgGpuTime;
			gpuStdev += temp*temp;
		}
		cpuStdev = sqrt( cpuStdev / (float) (nTrials-1) );
		gpuStdev = sqrt( gpuStdev / (float) (nTrials-1) );
		fprintf(timeFile,"%12.8f %12.8f %12.8f %12.8f", avgGpuTime, gpuStdev ,avgCpuTime, cpuStdev);
	
		// free host variables
		free(xyz_h);
		free(f_h); 
		free(ityp_h); 
		free(w_h); 
		free(g0_h); 
		free(gr2_h); 
		free(x0_h); 
		free(alpha_h); 
		free(lj_A_h); 
		free(lj_B_h); 
		// free device variables
		cudaFree(xyz_d); 
		cudaFree(f_d); 
		cudaFree(ityp_d); 
		cudaFree(w_d); 
		cudaFree(g0_d); 
		cudaFree(gr2_d); 
		cudaFree(x0_d); 
		cudaFree(alpha_d); 
		cudaFree(lj_A_d); 
		cudaFree(lj_B_d);

		fprintf(timeFile,"\n"); 
	}
	fclose(timeFile);
	return 0;

}


