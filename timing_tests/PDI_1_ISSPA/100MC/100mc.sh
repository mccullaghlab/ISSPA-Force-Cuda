#/bin/bash
rm ISSPA_force_100.xyz
touch ISSPA_force_100.xyz
export CUDA_VISIBLE_DEVICES=0
i=1
while [ $i -le 10000 ]
do
    j=$(printf "%06d" $i)
    rand=$RANDOM
    echo $rand
    cp ../config.txt .
     sed -i'' "s/XXX/"$rand"/g" config.txt
    ../../../src/total_force_cuda.x config.txt
    mv ISSPA_force_100.xyz test.xyz
    cat test.xyz ISSPA_force.xyz > ISSPA_force_100.xyz
    #mv ISSPA_force.xyz ISSPA_force_$j.xyz
    i=`expr $i + 1`
done
