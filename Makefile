
CUDA_FLAGS = -Wno-deprecated-gpu-targets -lcurand -use_fast_math
GNU_FLAGS = -I/usr/local/cuda/include -O3 -lcudart

CXX = g++
CUDA = nvcc

isspa: total_force_cuda.cpp isspa_force_cuda.cu isspa_force_cuda.h atom_class.cpp atom_class.h nonbond_cuda.cu nonbond_cuda.h config_class.h config_class.cpp leapfrog_cuda.h leapfrog_cuda.cu
	$(CXX) $(GNU_FLAGS) -c total_force_cuda.cpp atom_class.cpp config_class.cpp
	$(CUDA) $(CUDA_FLAGS) -c isspa_force_cuda.cu nonbond_cuda.cu leapfrog_cuda.cu
	$(CUDA) $(CUDA_FLAGS) total_force_cuda.o atom_class.o config_class.o isspa_force_cuda.o nonbond_cuda.o leapfrog_cuda.o -o total_force_cuda.x 


omp: isspa_force.omp.c
	g++ isspa_force.omp.c -o isspa_force.omp.x -O3 -fopenmp -use_fast_math
time: timing.cu
	nvcc timing.cu -o timing.x -lcurand -use_fast_math
