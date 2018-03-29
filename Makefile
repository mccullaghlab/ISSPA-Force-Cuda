

all: gen_parab_sphere.cu
	nvcc gen_parab_sphere.cu -o gen_parab_sphere.x -lcurand
