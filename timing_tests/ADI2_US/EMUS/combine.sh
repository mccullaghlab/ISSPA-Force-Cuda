#/bin/bash
export CUDA_VISIBLE_DEVICES=0

run=run1
for fp in $(seq 3.5 0.5 16.0)
do
    #cat ../window_$fp/ADI2.run0.window.$fp.dat  ../window_$fp/ADI2.run1.window.$fp.dat > ADI2.window.$fp.dat
    cat ../window_$fp/ADI2.run0.window.$fp.dat  ../window_$fp/ADI2.run1.window.$fp.dat > ADI2.window.$fp.dat
done


