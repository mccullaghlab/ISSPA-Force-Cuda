

#include "timing_class.h"


void timing::initialize()
{
	// start device timers
	cudaEventCreate(&totalStart);
	cudaEventCreate(&totalStop);
	cudaEventRecord(totalStart);
	cudaEventCreate(&writeStart);
	cudaEventCreate(&writeStop);
	bondTime = 0.0f;
	angleTime = 0.0f;
	dihTime = 0.0f;
	nonbondTime = 0.0f;
	neighborListTime = 0.0f;
	writeTime = 0.0f;
	leapFrogTime = 0.0f;
	isspaTime = 0.0f;
	usTime = 0.0f;

}

void timing::startWriteTimer()
{
	cudaEventRecord(writeStart);
}
void timing::stopWriteTimer()
{
	cudaEventRecord(writeStop);
	cudaEventSynchronize(writeStop);
	cudaEventElapsedTime(&milliseconds,writeStart,writeStop);
	writeTime += milliseconds;
}

void timing::print_final(float elapsedns)
{

	char *unit;
	float divider;

	// get GPU time
	cudaEventRecord(totalStop);
    	cudaEventSynchronize(totalStop);
	cudaEventElapsedTime(&milliseconds, totalStart, totalStop);
	// get smart unit of time
	divider = 1.0;
	unit = "ms";
	if (milliseconds > 1.0E3) {
		unit = "s";
		divider  = 1.0E3;
	}
	printf("Elapsed CPU/GPU time = %10.2f %s\n", milliseconds/divider, unit);
	printf("Simulation time = %10.2f ns\n", elapsedns);
	day_per_millisecond = 1e-3 /60.0/60.0/24.0;
	printf("Average ns/day = %10.2f\n", elapsedns/(milliseconds*day_per_millisecond) );
	
	printf("Bond force calculation time = %10.2f %s (%5.1f %%)\n", bondTime/divider, unit, bondTime/milliseconds*100);
	printf("Angle force calculation time = %10.2f %s (%5.1f %%)\n", angleTime/divider, unit, angleTime/milliseconds*100);
	printf("Dihedral force calculation time = %10.2f %s (%5.1f %%)\n", dihTime/divider, unit, dihTime/milliseconds*100);
	printf("Nonbond force calculation time = %10.2f %s (%5.1f %%)\n", nonbondTime/divider, unit, nonbondTime/milliseconds*100);
	printf("Neighbor list calculation time = %10.2f %s (%5.1f %%)\n", neighborListTime/divider, unit, neighborListTime/milliseconds*100);
	printf("US bias force calculation time = %10.2f %s (%5.1f %%)\n", usTime/divider, unit, usTime/milliseconds*100);
	printf("IS-SPA force calculation time = %10.2f %s (%5.1f %%)\n", isspaTime/divider, unit, isspaTime/milliseconds*100);
	printf("Leap-frog propogation time = %10.2f %s (%5.1f %%)\n", leapFrogTime/divider, unit, leapFrogTime/milliseconds*100);
	printf("Write trajectory file time = %10.2f %s (%5.1f %%)\n", writeTime/divider, unit, writeTime/milliseconds*100);
	printf("Total percent accounted for = %10.2f %%\n", (bondTime+angleTime+dihTime+nonbondTime+neighborListTime+usTime+isspaTime+leapFrogTime+writeTime)/milliseconds*100);


}

