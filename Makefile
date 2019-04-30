
CUDA_FLAGS = -Wno-deprecated-gpu-targets -lcurand -use_fast_math
GNU_FLAGS = -I/usr/local/cuda/include -O3 -lcudart

CXX = g++
CUDA = nvcc

isspa: total_force_cuda.cpp isspa_force_cuda.cu isspa_force_cuda.h read_prmtop.h read_prmtop.cpp constants.h dih_class.cpp dih_class.h angle_class.cpp angle_class.h bond_class.cpp bond_class.h atom_class.cpp atom_class.h nonbond_cuda.cu nonbond_cuda.h config_class.h config_class.cpp leapfrog_cuda.h leapfrog_cuda.cu neighborlist_cuda.h neighborlist_cuda.cu stringlib.h stringlib.c bond_force_cuda.h bond_force_cuda.cu angle_force_cuda.h angle_force_cuda.cu dih_force_cuda.h dih_force_cuda.cu
	$(CXX) $(GNU_FLAGS) -c total_force_cuda.cpp atom_class.cpp config_class.cpp bond_class.cpp angle_class.cpp dih_class.cpp read_prmtop.cpp stringlib.c
	$(CUDA) $(CUDA_FLAGS) -c isspa_force_cuda.cu nonbond_cuda.cu leapfrog_cuda.cu neighborlist_cuda.cu bond_force_cuda.cu angle_force_cuda.cu dih_force_cuda.cu
	$(CUDA) $(CUDA_FLAGS) total_force_cuda.o read_prmtop.o atom_class.o config_class.o dih_class.o angle_class.o bond_class.o isspa_force_cuda.o nonbond_cuda.o leapfrog_cuda.o neighborlist_cuda.o stringlib.o bond_force_cuda.o angle_force_cuda.o dih_force_cuda.o -o total_force_cuda.x 


omp: isspa_force.omp.c
	g++ isspa_force.omp.c -o isspa_force.omp.x -O3 -fopenmp -use_fast_math
time: timing.cu
	nvcc timing.cu -o timing.x -lcurand -use_fast_math
