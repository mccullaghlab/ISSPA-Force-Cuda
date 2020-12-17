#/bin/bash
export CUDA_VISIBLE_DEVICES=0

run=run0
for fp in $(seq 3.5 0.5 25.0)
do
     echo $fp
     sed -e "s/XXX/$fp/g" LJ2.$run.us > us_params.$run.dat
     sed -e "s/XXX/$fp/g" ${run}_config.txt > ${run}_config.dat
     mv us_params.$run.dat window_$fp/
     mv ${run}_config.dat window_$fp/     
     cp submit_${run}.sh window_$fp/
     cd window_$fp/
     bash submit_$run.sh
     cd ../
done


