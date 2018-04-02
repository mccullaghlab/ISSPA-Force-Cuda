

isspa: total_force.c isspa_force.cu isspa_force.h
	nvcc -c total_force.c isspa_force.cu -o isspa_force.x -lcurand -use_fast_math
	nvcc total_force.o isspa_force.o -o isspa_force.x -lcurand -use_fast_math
omp: isspa_force.omp.c
	g++ isspa_force.omp.c -o isspa_force.omp.x -O3 -fopenmp -use_fast_math
omp: isspa_force.omp.c
	g++ isspa_force.omp.c -o isspa_force.omp.x -O3 -fopenmp -use_fast_math
time: timing.cu
	nvcc timing.cu -o timing.x -lcurand -use_fast_math
