#!/bin/bash
#type=LJsolu
#type=Csolu
#type=Csolv_pair
#type=Csolv
#type=LJsolv
type=Csolv_CDD
rm ISSPA_force_${type}_10000.xyz
touch ISSPA_force_${type}_10000.xyz
rm random_number_seed.dat
tough random_number_seed.dat
export CUDA_VISIBLE_DEVICES=0
i=1
while [ $i -le 1000 ]
do
    j=$(printf "%06d" $i)
    rand=$RANDOM
    echo $rand >> random_number_seed.dat
    cp ../config.txt .
     sed -i'' "s/XXX/"$rand"/g" config.txt
    ../../src/total_force_cuda.x config.txt
    mv ISSPA_force_${type}_10000.xyz test.xyz
    cat test.xyz ISSPA_force.xyz > ISSPA_force_${type}_10000.xyz
    i=`expr $i + 1`
done
