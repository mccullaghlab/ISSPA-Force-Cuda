
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "constants.h"
//#include "constants_cuda.h"

class config
{
	private:
		int i;
		FILE *configFile;
	public:
		float T;    // temperature
		float dt;   // integration timestep in amber units
		float dtPs;   // integration timestep in ps
		float pnu;  // Anderson thermostat frequency	
		int nSteps;
		int deltaWrite;
		int deltaNN;
		int nMC;    // number of Monte Carlo points for solvent
		int seed;
		int us;
		int velRst;
		float lbox;
		float eps;
		float rCut,rCut2;
		float rNN,rNN2;
		char prmtopFileName[MAXCHAR];
		char isspaPrmtopFileName[MAXCHAR];
		char inputCoordFileName[MAXCHAR];
		char inputVelFileName[MAXCHAR];
		char usCfgFileName[MAXCHAR];
		char forOutFileName[MAXCHAR];
		char posOutFileName[MAXCHAR];
		char velOutFileName[MAXCHAR];
		char posRstFileName[MAXCHAR];
		char velRstFileName[MAXCHAR];
		// read config file 
		void initialize(char *);
		// cuda stuff
		void set_cuda_constants();
		
};
