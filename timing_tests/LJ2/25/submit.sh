
rm random_number_seed.dat
touch random_number_seed.dat
for i in $(seq 43.3 0.1 50.0)
do
    echo $i    
    rand=$RANDOM
    echo $rand >> random_number_seed.dat
    sed -e "s|XXX|$i|g" -e "s/YYY/"$rand"/g" config.txt > config_$i.txt
    sed -e "s|XXX|$i|g" test.sh > submit_$i.sh
    mv config_$i.txt $i/
    mv submit_$i.sh $i/
    cd $i/
    bash submit_$i.sh
    cd ../
done
