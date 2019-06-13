

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

	// get GPU time
	cudaEventRecord(totalStop);
    	cudaEventSynchronize(totalStop);
	cudaEventElapsedTime(&milliseconds, totalStart, totalStop);
	printf("Elapsed CPU/GPU time = %10.2f ms\n", milliseconds);
	printf("Simulation time = %10.2f ns\n", elapsedns);
	day_per_millisecond = 1e-3 /60.0/60.0/24.0;
	printf("Average ns/day = %10.2f\n", elapsedns/(milliseconds*day_per_millisecond) );
	
	printf("Bond force calculation time = %10.2f ms (%5.1f %%)\n", bondTime, bondTime/milliseconds*100);
	printf("Angle force calculation time = %10.2f ms (%5.1f %%)\n", angleTime, angleTime/milliseconds*100);
	printf("Dihedral force calculation time = %10.2f ms (%5.1f %%)\n", dihTime, dihTime/milliseconds*100);
	printf("Nonbond force calculation time = %10.2f ms (%5.1f %%)\n", nonbondTime, nonbondTime/milliseconds*100);
	printf("Neighbor list calculation time = %10.2f ms (%5.1f %%)\n", neighborListTime, neighborListTime/milliseconds*100);
	printf("IS-SPA time = %10.2f ms (%5.1f %%)\n", isspaTime, isspaTime/milliseconds*100);
	printf("Leap-frog propogation time = %10.2f ms (%5.1f %%)\n", leapFrogTime, leapFrogTime/milliseconds*100);
	printf("Write trajectory file time = %10.2f ms (%5.1f %%)\n", writeTime, writeTime/milliseconds*100);


}

