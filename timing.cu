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

//MM
__global__ void nonbond_kernel(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom1;
	int atom2;
	int it;    // atom type of atom of interest
	int jt;    // atom type of other atom
	float temp, dist2;	
	int i, k;
	int count;
	float r[3];
	float r2, r6, fs;
	float hbox;

	if (index < nAtoms*(nAtoms-1)/2)
	{
		hbox = lbox/2.0;
		// determine two atoms to work on based on recursive definition
		count = 0;
		for (i=0;i<nAtoms-1;i++) {
			count += nAtoms-1-i;
			if (index < count) {
				atom1 = i;	
				atom2 = nAtoms - count + index;
				break;
			}
		}
		// get interaction type
		it = ityp[atom1];
		jt = ityp[atom2];
		dist2 = 0.0f;
		for (k=0;k<nDim;k++) {
			r[k] = xyz[atom1*nDim+k] - xyz[atom2*nDim+k];
			if (r[k] > hbox) {
				r[k] -= (int)(temp/hbox) * lbox;
			} else if (r[k] < -hbox) {
				r[k] += (int)(temp/hbox) * lbox;
			}
			dist2 += r[k]*r[k];
		}
		// LJ force
		r2 = 1/dist2;
		r6 = r2 * r2 * r2;
		fs = r6 * (lj_B[it] - lj_A[it] * r6);
		atomicAdd(&f[atom1*nDim], fs*r[0] );
		atomicAdd(&f[atom1*nDim+1], fs*r[1] );
		atomicAdd(&f[atom1*nDim+2], fs*r[2] );
		atomicAdd(&f[atom2*nDim], -fs*r[0] );
		atomicAdd(&f[atom2*nDim+1], -fs*r[1] );
		atomicAdd(&f[atom2*nDim+2], -fs*r[2] );

	}
}


__global__ void nonbond_kernel_natoms(float *xyz, float *f, float *charges, float *lj_A, float *lj_B, int *ityp, int nAtoms, float lbox) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int atom1;
	int atom2;
	int it;    // atom type of atom of interest
	int jt;    // atom type of other atom
	float temp, dist2;	
	int i, k;
	int count;
	float r[3];
	float r2, r6, fs;
	float hbox;

	if (index < nAtoms)
	{
		hbox = lbox/2.0;
		// determine two atoms to work on based on recursive definition
		atom1 = index;
		for (atom2=0;atom2<nAtoms;atom2++) {
			if (atom2 != atom1) {
				// get interaction type
				it = ityp[atom1];
				jt = ityp[atom2];
				dist2 = 0.0f;
				for (k=0;k<nDim;k++) {
					r[k] = xyz[atom1*nDim+k] - xyz[atom2*nDim+k];
					if (r[k] > hbox) {
						r[k] -= (int)(temp/hbox) * lbox;
					} else if (r[k] < -hbox) {
						r[k] += (int)(temp/hbox) * lbox;
					}
					dist2 += r[k]*r[k];
				}
				// LJ force
				r2 = 1/dist2;
				r6 = r2 * r2 * r2;
				fs = r6 * (lj_B[it] - lj_A[it] * r6);
				f[atom1*nDim] += fs*r[0];
				f[atom1*nDim+1] += fs*r[1];
				f[atom1*nDim+2] += fs*r[2];
//				atomicAdd(&f[atom2*nDim], -fs*r[0] );
//				atomicAdd(&f[atom2*nDim+1], -fs*r[1] );
//				atomicAdd(&f[atom2*nDim+2], -fs*r[2] );
			}
		}

	}
}

//MM
__global__ void isspa_force_natoms_nmc(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC) {
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
//		curand_init(0,blockIdx.x,index,&state);
		curand_init(0,index,0,&state);
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

__global__ void isspa_force_natoms(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC) {
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
	int ev_flag, k, mc;
	float rinv, r6, fs;
	curandState_t state;

	if (index < nAtoms)
	{
		// get atom number of interest
		atom = index;
		it = ityp[atom];
		// initialize random number generator
//		curand_init(0,blockIdx.x,index,&state);
		curand_init(0,index,0,&state);
		for (mc=0;mc<nMC;mc++) {
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
				f[atom*nDim] += fs*mc_pos[0];
				f[atom*nDim+1] += fs*mc_pos[1];
				f[atom*nDim+2] += fs*mc_pos[2];
			}
		}
	}
}

//MM

void cpu_isspa_force(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC) {
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
	int nMaxAtoms = 100000;
	int nAtoms;
	int nTrials = 10;
	float gpu2Time[nTrials];
	float gpu1Time[nTrials];
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
	float *charges_h;  // alpha parameter for g - host data
	float *charges_d;  // alpha parameter for g - device data
	float *lj_A_h;   // Lennard-Jones A parameter - host data
	float *lj_A_d;   // Lennard-Jones A parameter - device data
	float *lj_B_h;   // Lennard-Jones B parameter - host data
	float *lj_B_d;   // Lennard-Jones B parameter - device data
	int nMC = MC;    // number of MC points
	int nAtomBytes, nTypeBytes, i;
	cudaEvent_t start, stop;
	float milliseconds;
	float avgGpu1Time,avgGpu2Time;
	float gpu1Stdev, gpu2Stdev, temp;
	unsigned int long seed = 12345;
	float lbox;

	timeFile = fopen("timing.dat", "w");	

	for (nAtoms=1000;nAtoms<=nMaxAtoms;nAtoms += 1000){
		fprintf(timeFile,"%20d",nAtoms);
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
		charges_h = (float *)malloc(nAtomBytes);
		cudaMalloc((void **) &charges_d, nAtomBytes);
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
			charges_h[i] = 0.0;
		}
		lbox = (float) nAtoms * 8.0;
		lj_A_h[0] = 6.669e7;
		lj_B_h[0] = 1.103e4;

		avgGpu1Time = avgGpu2Time = 0.0;
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
			cudaMemcpy(charges_d, charges_h, nAtomBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(lj_A_d, lj_A_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(lj_B_d, lj_B_h, nTypeBytes, cudaMemcpyHostToDevice);	
			
			// determine gridSize and blockSize
			//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nonbond_kernel, 0, nAtoms*(nAtoms-1)/2); 
			
			// Round up according to array size 
			//gridSize = (nAtoms*(nAtoms-1)/2 + blockSize - 1) / blockSize; 
			
			// run parabola random cuda kernal
			//nonbond_kernel<<<gridSize, blockSize>>>(xyz_d, f_d, charges_d, lj_A_d, lj_B_d, ityp_d, nAtoms, lbox);

			// pass device variable, f_d, to host variable f_h
			cudaMemcpy(f_h, f_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
			
			// get GPU time
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			gpu1Time[trial] = milliseconds;
			avgGpu1Time += milliseconds;
			// rezero forces
			for (i=0;i<nAtoms;i++) {
				f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
			}

			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
			
			// copy data to device
			cudaMemcpy(f_d, f_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
			cudaMemcpy(xyz_d, xyz_h, nAtomBytes*nDim, cudaMemcpyHostToDevice);	
			cudaMemcpy(ityp_d, ityp_h, nAtoms*sizeof(int), cudaMemcpyHostToDevice);	
			cudaMemcpy(charges_d, charges_h, nAtomBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(lj_A_d, lj_A_h, nTypeBytes, cudaMemcpyHostToDevice);	
			cudaMemcpy(lj_B_d, lj_B_h, nTypeBytes, cudaMemcpyHostToDevice);	
			
			// determine gridSize and blockSize
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, isspa_force_natoms_nmc, 0, nAtoms); 
			
			// Round up according to array size 
			gridSize = (nAtoms + blockSize - 1) / blockSize; 
			
			//printf("gridSize = %d, blockSize = %d\n", gridSize, blockSize);
			// run parabola random cuda kernal
			nonbond_kernel_natoms<<<gridSize, blockSize>>>(xyz_d, f_d, charges_d, lj_A_d, lj_B_d, ityp_d, nAtoms, lbox);
			
			// pass device variable, f_d, to host variable f_h
			cudaMemcpy(f_h, f_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	

			// get GPU time
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			gpu2Time[trial] = milliseconds;
			avgGpu2Time += milliseconds;

//			fprintf(timeFile,"%12.8f", milliseconds);
		}
		temp = 0.0;
		avgGpu1Time /= (float) nTrials;
		avgGpu2Time /= (float) nTrials;
		gpu1Stdev = 0.0;
		gpu2Stdev = 0.0;
		for (trial=0;trial<nTrials;trial++) {
			temp = gpu1Time[trial] - avgGpu1Time;
			gpu1Stdev += temp*temp;
			temp = gpu2Time[trial] - avgGpu2Time;
			gpu2Stdev += temp*temp;
		}
		gpu1Stdev = sqrt( gpu1Stdev / (float) (nTrials-1) );
		gpu2Stdev = sqrt( gpu2Stdev / (float) (nTrials-1) );
		fprintf(timeFile,"%20.8f %20.8f %20.8f %20.8f", avgGpu1Time, gpu1Stdev ,avgGpu2Time, gpu2Stdev);
	
		// free host variables
		free(xyz_h);
		free(f_h); 
		free(ityp_h); 
		free(charges_h); 
		free(lj_A_h); 
		free(lj_B_h); 
		// free device variables
		cudaFree(xyz_d); 
		cudaFree(f_d); 
		cudaFree(ityp_d); 
		cudaFree(charges_d); 
		cudaFree(lj_A_d); 
		cudaFree(lj_B_d);

		fprintf(timeFile,"\n"); 
		fflush(timeFile);
	}
	fclose(timeFile);
	return 0;

}


