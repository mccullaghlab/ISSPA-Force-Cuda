#/bin/bash
export CUDA_VISIBLE_DEVICES=0

j=8
while [ $j -le 152 ]
do
    i=1
    rm tscaling.MC$j.dat
    touch tscaling.MC$j.dat
    while [ $i -le 100 ]
    do
        rand=$RANDOM
        sed -e "s/XXX/"$rand"/g" -e "s/YYY/"$j"/g" config.txt > test.txt
        ../bin/total_force_cuda.x test.txt > test.log
        awk 'FNR==58' test.log | awk ' {print $4}' >>  tscaling.MC$j.dat 
        i=`expr $i + 1`
    done
    j=`expr $j + 8`
done



