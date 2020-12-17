#/bin/bash
export CUDA_VISIBLE_DEVICES=0

for fp in $(seq 4.0 0.5 25.0)
do
    cat ../window_$fp/LJ2.run0.window.$fp.dat > LJ2.window.$fp.dat
    #cat ../window_$fp/LJ2.run0.window.$fp.dat  ../window_$fp/LJ2.run1.window.$fp.dat > LJ2.window.$fp.dat
done


