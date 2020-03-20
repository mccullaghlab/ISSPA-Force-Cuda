#/bin/bash
export CUDA_VISIBLE_DEVICES=0
j=1
while [ $j -le 20 ]
do
      i=1
      rm ISSPA_force_10_$j.xyz
      touch ISSPA_force_10_$j.xyz
      while [ $i -le 5000 ]
      do
	  rand=$RANDOM
	  echo $rand
	  cp ../config.txt .
	  sed -i'' "s/XXX/"$rand"/g" config.txt
	  ../../../src/total_force_cuda.x config.txt
	  mv ISSPA_force_10_$j.xyz test.xyz
	  cat test.xyz ISSPA_force.xyz > ISSPA_force_10_$j.xyz
	  i=`expr $i + 1`
      done
      j=`expr $j + 1`
done
