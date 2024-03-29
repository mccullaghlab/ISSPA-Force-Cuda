#/bin/bash
export CUDA_VISIBLE_DEVICES=0

prev=3.5
for fp in $(seq 4.0 0.5 16.0)
do
     echo $fp
     mkdir window_$fp
     sed -e "s/XXX/$fp/g" ADI2.us > us_params.dat
     sed -e "s/XXX/$fp/g"  -e "s/YYY/$prev/g" equil_config.txt > equil_config.dat
     mv us_params.dat window_$fp/
     mv equil_config.dat window_$fp/     
     cp submit.sh window_$fp/
     cd window_$fp/
     bash submit.sh
     cd ../
     prev=$fp
done


