#/bin/bash
export CUDA_VISIBLE_DEVICES=0

../../../bin/total_force_cuda.x run0_config.dat > LJ2.run0.log
