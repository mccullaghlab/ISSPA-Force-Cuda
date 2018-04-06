
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define nDim 3

class config
{
	private:
		int i;
		FILE *configFile;
	public:
		float T;    // temperature
		float dt;   // integration timestep
		float pnu;  // Anderson thermostat frequency	
		int nSteps;
		int deltaWrite;
		int deltaNN;
		float lbox;
		float rcut;
		float rNN;
		// initialize all variables
		void initialize();
		
};
