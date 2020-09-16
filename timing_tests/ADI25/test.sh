
#/bin/bash
export CUDA_VISIBLE_DEVICES=0
#/usr/local/cuda-9.2/bin/cuda-memcheck --log-file memcheck.log ../../bin/total_force_cuda.x config.txt  
../../bin/total_force_cuda.x config.txt
#../../bin/total_force_cuda.x config.txt > PDI.2.run02.log
  
