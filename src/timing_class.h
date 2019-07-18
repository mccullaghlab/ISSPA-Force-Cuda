
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

class timing
{


	public:
		cudaEvent_t totalStart, totalStop;
		cudaEvent_t writeStart, writeStop;
		float bondTime;
		float angleTime;
		float dihTime;
		float nonbondTime;
		float neighborListTime;
		float leapFrogTime;
		float isspaTime;
		float usTime;
		float writeTime;
		float milliseconds;
		float day_per_millisecond;

		void initialize();
		void startWriteTimer();
		void stopWriteTimer();
		void print_final(float );		
};
