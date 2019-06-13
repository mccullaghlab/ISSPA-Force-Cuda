
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define nDim 3
#define MC 10

//MM
void cpu_ispa_force(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC, int nThreads) {
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

#pragma omp parallel for num_threads(nThreads) \
	shared (f, xyz, ityp, nAtoms, nMC, alpha, x0, w, g0, gr2, lj_A, lj_B) \
	private (atom, atom2, index, it, jt, mc_pos, ev_flag, k, rinv, r6, fs, gnow, rnow, r2, x1, x2, mc_pos_atom, prob, attempt,dist2,temp)
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
		r2 = 2.0f * sqrt(1.0f - r2);
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
					temp = sqrt(dist2)-x0[jt];
					gnow *= (-alpha[jt] * temp*temp + g0[jt]);
				}
			}
		}
		
		if (ev_flag ==0) {
			rinv = 1.0f / rnow;
			r2 = rinv * rinv;
			r6 = r2 * r2 * r2;
			fs = gnow * r6 * (lj_B[it] - lj_A[it] * r6);
#pragma omp atomic
			f[atom*nDim] += fs*mc_pos[0];
#pragma omp atomic
			f[atom*nDim+1] += fs*mc_pos[1];
#pragma omp atomic
			f[atom*nDim+2] += fs*mc_pos[2];
		}

	}
}
//MM

int main(void)  
{
	FILE *timeFile;
	FILE *xyzFile;
	int nAtoms = 1000;
	int nAtomTypes = 1;
	int nTrials = 10;
	float cpuTime[nTrials];
	int trial;
	int maxThreads = 10;
	int thread;
	float *xyz_h;    // coordinate array - host data
	float *f_h;      // force array - host data
	int *ityp_h;     // atom type array - host data
	float *x0_h;     // center position of parabola and g - host data 
	float *g0_h;     // height of parabola approximation of g - host data 
	float *gr2_h;     // excluded volume distance and end of parabola distance squared - host data 
	float *w_h;      // width of parabola - host data
	float *alpha_h;  // alpha parameter for g - host data
	float *lj_A_h;   // Lennard-Jones A parameter - host data
	float *lj_B_h;   // Lennard-Jones B parameter - host data
	int nMC = MC;    // number of MC points
	int nAtomBytes, nTypeBytes, i;
	float milliseconds;
	unsigned int long seed = 12345;
	float avgCpuTime;
	float stdevCpuTime, temp;
	double startTime, stopTime;

	// size of xyz arrays
	nAtomBytes = nAtoms*sizeof(float);
	nTypeBytes = nAtomTypes*sizeof(float);
 	// allocate atom coordinate arrays
	xyz_h = (float *)malloc(nAtomBytes*nDim);
 	// allocate atom force arrays
	f_h = (float *)malloc(nAtomBytes*nDim);
 	// allocate atom type arrays
	ityp_h = (int *)malloc(nAtoms*sizeof(int));
	// allocate atom based parameter arrays
	x0_h = (float *)malloc(nTypeBytes);
	g0_h = (float *)malloc(nTypeBytes);
	gr2_h = (float *)malloc(nTypeBytes*2);
	w_h = (float *)malloc(nTypeBytes);
	alpha_h = (float *)malloc(nTypeBytes);
	lj_A_h = (float *)malloc(nTypeBytes);
	lj_B_h = (float *)malloc(nTypeBytes);


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

	timeFile = fopen("timing.dat", "w");	
	
	for (thread=0;thread<maxThreads;thread++) {
		avgCpuTime = 0;
		for (trial=0;trial<nTrials;trial++) {
			// set force to zero
			for (i=0;i<nAtoms;i++) {
				f_h[i*nDim] = f_h[i*nDim+1] = f_h[i*nDim+2] = 0.0f;
			}
			// start device timer
			startTime = omp_get_wtime();	

			cpu_ispa_force(xyz_h, f_h, w_h, x0_h, g0_h, gr2_h, alpha_h, lj_A_h, lj_B_h, ityp_h, nAtoms, nMC,thread+1);

			// stop device timer
			stopTime = omp_get_wtime();
			cpuTime[trial] = stopTime - startTime;
			avgCpuTime += cpuTime[trial];
		}
		avgCpuTime /= (float) nTrials;
		stdevCpuTime = 0.0;
		for (trial=0;trial<nTrials;trial++) {
			temp = cpuTime[trial] - avgCpuTime;
			stdevCpuTime += temp*temp;
		}
		stdevCpuTime = sqrt( stdevCpuTime / (float) (nTrials-1));
		fprintf(timeFile, "%10d %15.10f %15.10f\n", thread+1, avgCpuTime, stdevCpuTime);
	}
	
				
	printf("Time = %12.8f ms\n", stopTime - startTime);

	// print xyz file
	xyzFile = fopen("forces.xyz","w");
	fprintf(xyzFile,"%d\n", nAtoms);
	fprintf(xyzFile,"%d\n", nAtoms);
	for (i=0;i<nAtoms; i++) 
	{
		fprintf(xyzFile,"C %10.6f %10.6f %10.6f\n", f_h[i*nDim]/((float) nMC),f_h[i*nDim+1]/((float) nMC),f_h[i*nDim+2]/((float) nMC));
	}
	fclose(xyzFile);
/*	// print xyz file
	mcXyzFile = fopen("mc_points.xyz","w");
	fprintf(mcXyzFile,"%d\n", nAtoms*nMC);
	fprintf(mcXyzFile,"%d\n", nAtoms*nMC);
	for (i=0;i<nAtoms*nMC; i++) 
	{
		fprintf(mcXyzFile,"C %10.6f %10.6f %10.6f\n", mc_xyz_h[i*nDim],mc_xyz_h[i*nDim+1],mc_xyz_h[i*nDim+2]);
	}
	fclose(mcXyzFile);
*/
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

	return 0;

}


