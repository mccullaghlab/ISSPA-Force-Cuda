#/bin/bash
export CUDA_VISIBLE_DEVICES=0

prev=3.5
for fp in $(seq 9.0 0.5 16.0)
do
     echo $fp
     mkdir window_$fp
     sed -e "s/XXX/$fp/g" ADI2.us > us_params.dat
     sed -e "s/XXX/$fp/g"  -e "s/YYY/$prev/g" config.txt > config.dat
     mv us_params.dat window_$fp/
     mv config.dat window_$fp/     
     cp submit.sh window_$fp/
     cd window_$fp/
     bash submit.sh
     cd ../
     prev=$fp
done


