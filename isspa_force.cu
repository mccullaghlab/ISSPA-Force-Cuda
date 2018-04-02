#include <isspa_force.h>

//__global__ void init_rand(unsigned int long seed, curandState_t* states){
//	curand_init(seed,blockIdx.x,0,&states);
//}

__global__ void isspa_force(float *xyz, float *f, float *w, float *x0, float *g0, float *gr2, float *alpha, float *lj_A, float *lj_B, int *ityp, int nAtoms, int nMC) {
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
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
	int ev_flag, k;
	float rinv, r6, fs;
	curandState_t state;

	if (index < nAtoms*nMC)
	{
		// get atom number of interest
		atom = index%nAtoms;
		it = ityp[atom];
		// initialize random number generator
		curand_init(0,blockIdx.x,index,&state);
		// select one point from 1D parabolic distribution
		rnow = 1.0f - 2.0f * curand_uniform(&state);
		prob = rnow*rnow;
		attempt = curand_uniform(&state);
		while (attempt < prob)
		{
			rnow = 1.0f - 2.0f * curand_uniform(&state);
			prob = rnow*rnow;
			attempt = curand_uniform(&state);
		}
		rnow = w[it] * rnow + x0[it];
		// select two points on surface of sphere
		x1 = 1.0f - 2.0f * curand_uniform(&state);
		x2 = 1.0f - 2.0f * curand_uniform(&state);
		r2 = x1*x1 + x2*x2;
		while (r2 > 1.0f) 
		{
			x1 = 1.0f - 2.0f * curand_uniform(&state);
                	x2 = 1.0f - 2.0f * curand_uniform(&state);
			r2 = x1*x1 + x2*x2;
		}
		// generate 3D MC pos based on position on surface of sphere and parabolic distribution in depth
		mc_pos[0] = rnow*(1.0f - 2.0f*r2);
		r2 = 2.0f * sqrtf(1.0f - r2);
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
					temp = sqrtf(dist2)-x0[jt];
					gnow *= (-alpha[jt] * temp*temp + g0[jt]);
				}
			}
		}
		
		if (ev_flag ==0) {
			rinv = 1.0f / rnow;
			r2 = rinv * rinv;
			r6 = r2 * r2 * r2;
			fs = gnow * r6 * (lj_B[it] - lj_A[it] * r6);
			atomicAdd(&f[atom*nDim], fs*mc_pos[0]);
			atomicAdd(&f[atom*nDim+1], fs*mc_pos[1]);
			atomicAdd(&f[atom*nDim+2], fs*mc_pos[2]);
		}

	}
}


