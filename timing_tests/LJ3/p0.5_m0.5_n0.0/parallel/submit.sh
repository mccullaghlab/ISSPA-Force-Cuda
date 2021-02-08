rm random_number_seed.dat
touch random_number_seed.dat
#for i in $(seq 5.7 0.1 15.0)
list=( 3.5 5.0 7.0 10.0 12.0 15.0 )
for i in "${list[@]}"
do
    for j in $(seq 3.5 0.1 50.0)
    do
        echo $i $j   
        rand=$RANDOM
        echo $rand >> random_number_seed.dat
        sed -e "s|XXX|$i|g" -e "s|YYY|$j|g" -e "s/ZZZ/"$rand"/g" config.txt > config_${i}_$j.txt
        sed -e "s|XXX|$i|g" -e "s|YYY|$j|g" test.sh > submit_${i}_$j.sh
        mv config_${i}_$j.txt $i/$j/
        mv submit_${i}_$j.sh $i/$j/
        cd $i/$j/
        bash submit_${i}_$j.sh
        cd ../../
    done
done
