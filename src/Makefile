
CUDA_FLAGS = -Wno-deprecated-gpu-targets -Wno-deprecated-declarations -lcurand -use_fast_math -arch=sm_52 -lineinfo
GNU_FLAGS = -I/usr/local/cuda/include -O3 -lcudart

CXX = g++
CUDA = nvcc

isspa: total_force_cuda.cpp isspa_force_cuda.cu isspa_force_cuda.h read_prmtop.h read_prmtop.cpp constants.h isspa_class.cpp dih_class.cpp dih_class.h angle_class.cpp angle_class.h bond_class.cpp bond_class.h atom_class.cpp atom_class.h nonbond_force_cuda.cu nonbond_force_cuda.h config_class.h config_class.cpp leapfrog_cuda.h leapfrog_cuda.cu neighborlist_cuda.h neighborlist_cuda.cu stringlib.h stringlib.c bond_force_cuda.h bond_force_cuda.cu angle_force_cuda.h angle_force_cuda.cu dih_force_cuda.h dih_force_cuda.cu init_rand.cuh init_rand.cu timing_class.h timing_class.cpp cuda_vector_routines.h us_class.h us_class.cpp us_force_cuda.h us_force_cuda.cu
	$(CXX) $(GNU_FLAGS) -c total_force_cuda.cpp atom_class.cpp config_class.cpp bond_class.cpp angle_class.cpp dih_class.cpp isspa_class.cpp read_prmtop.cpp us_class.cpp timing_class.cpp stringlib.c
	$(CUDA) $(CUDA_FLAGS) -c isspa_force_cuda.cu nonbond_force_cuda.cu leapfrog_cuda.cu neighborlist_cuda.cu bond_force_cuda.cu angle_force_cuda.cu dih_force_cuda.cu us_force_cuda.cu init_rand.cu
	$(CUDA) $(CUDA_FLAGS) total_force_cuda.o read_prmtop.o atom_class.o timing_class.o config_class.o dih_class.o angle_class.o bond_class.o isspa_class.o us_class.o isspa_force_cuda.o nonbond_force_cuda.o leapfrog_cuda.o neighborlist_cuda.o us_force_cuda.o stringlib.o bond_force_cuda.o angle_force_cuda.o dih_force_cuda.o init_rand.o -o total_force_cuda.x 
	cp total_force_cuda.x ../bin/
	#cp total_force_cuda.x ~/bin/

