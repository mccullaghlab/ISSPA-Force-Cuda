
#/bin/bash
export CUDA_VISIBLE_DEVICES=0
../bin/total_force_cuda.x config.txt
#../../bin/total_force_cuda.x config.txt > $1_run01.dat
