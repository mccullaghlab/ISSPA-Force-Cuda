
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

class timing
{


	public:
		cudaEvent_t totalStart, totalStop;
		cudaEvent_t bondStart, bondStop;
		float bondTime;
		float angleTime;
		float dihTime;
		cudaEvent_t nonbondStart, nonbondStop;
		float nonbondTime;
		cudaEvent_t neighborListStart, neighborListStop;
		float neighborListTime;
		cudaEvent_t leapFrogStart, leapFrogStop;
		float leapFrogTime;
		float milliseconds;
		float day_per_millisecond;

		void initialize();
		void print_final(float );		
};
