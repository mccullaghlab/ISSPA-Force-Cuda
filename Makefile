

isspa: isspa_force.cu
	nvcc isspa_force.cu -o isspa_force.x -lcurand -use_fast_math
time: timing.cu
	nvcc timing.cu -o timing.x -lcurand -use_fast_math
