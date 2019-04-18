
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "constants.h"

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
		int nMC;    // number of Monte Carlo points for solvent
		float lbox;
		float rcut,rcut2;
		float rNN,rNN2;
		char prmtopFileName[MAXCHAR];
		char inputCoordFileName[MAXCHAR];
		// initialize all variables
		void initialize(char *);
		
};
