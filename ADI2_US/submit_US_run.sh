#/bin/bash
export CUDA_VISIBLE_DEVICES=0

run=run3
for fp in $(seq 4.0 0.5 11.0)
do
     echo $fp
     sed -e "s/XXX/$fp/g" ADI2.$run.us > us_params.$run.dat
     sed -e "s/XXX/$fp/g" ${run}_config.txt > ${run}_config.dat
     mv us_params.$run.dat window_$fp/
     mv ${run}_config.dat window_$fp/     
     cp submit_${run}.sh window_$fp/
     cd window_$fp/
     bash submit_$run.sh
     cd ../
done


