#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
../../../bin/total_force_cuda.x equil_config.txt > ADI2.equil.log
  
