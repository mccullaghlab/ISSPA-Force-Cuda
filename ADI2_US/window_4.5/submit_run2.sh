#/bin/bash
export CUDA_VISIBLE_DEVICES=0


../../../bin/total_force_cuda.x run2_config.dat > ADI2.run2.log
