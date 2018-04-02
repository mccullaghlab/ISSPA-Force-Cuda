#include <isspa_force.h>

int main(void)  
{
	FILE *xyzFile;
	FILE *mcXyzFile;
	int nAtoms = 10;
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
	unsigned int long seed = 12345;

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
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, isspa_force, 0, nAtoms*nMC); 

    	// Round up according to array size 
    	gridSize = (nAtoms*nMC + blockSize - 1) / blockSize; 

	printf("gridSize = %d, blockSize = %d\n", gridSize, blockSize);
	// run parabola random cuda kernal
	isspa_force<<<gridSize, blockSize>>>(xyz_d, f_d, w_d, x0_d, g0_d, gr2_d, alpha_d, lj_A_d, lj_B_d, ityp_d, nAtoms, nMC);

	// pass device variable, f_d, to host variable f_h
	cudaMemcpy(f_h, f_d, nAtomBytes*nDim, cudaMemcpyDeviceToHost);	
//	cudaMemcpy(mc_xyz_h, mc_xyz_d, nAtomBytes*nDim*nMC, cudaMemcpyDeviceToHost);	

	// get GPU time
	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time = %12.8f ms\n", milliseconds);

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

	return 0;

}


