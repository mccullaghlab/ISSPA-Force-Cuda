#/bin/bash
export CUDA_VISIBLE_DEVICES=0

rm tscaling.ADI10.dat
touch  tscaling.ADI10.dat
rm random_number_seed.dat
touch random_number_seed.dat
echo $rand >> random_number_seed.dat
i=1
while [ $i -le 100 ]
do
    rand=$RANDOM
    sed -i'' "s/XXX/"$rand"/g" config.txt
    ../bin/total_force_cuda.x config.txt > test.log
    awk 'FNR==58' test.log | awk ' {print $4}' >>  tscaling.ADI10.dat 
    i=`expr $i + 1`
done



