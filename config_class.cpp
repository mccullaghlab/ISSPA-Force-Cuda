
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
#include "config_class.h"


void config::initialize()
{
	dt = 0.002*20.455;
	T = 298.0 * 0.00198717;
	pnu = 0.001f;
	nSteps = 100000;
	deltaWrite = 100;
	lbox = 200.0;
	deltaNN = 10;
	rcut = 12.0;
	rcut2 = rcut*rcut;
	rNN = 15.0;
	rNN2 = rNN*rNN;
	nMC = 10;
}

