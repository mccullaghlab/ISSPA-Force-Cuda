list=( 3.5 5.0 7.0 10.0 12.0 15.0 )
#for i in $(seq 3.5 0.1 15.0)
for i in "${list[@]}"
do
    for j in $(seq 3.5 0.1 50.0)
    do
        mkdir $i
        mkdir $i/$j
        sed -e "s|YYY|$i|g" -e "s|XXX|$j|g" LJ3.leap > test.leap
        tleap -f test.leap
        mv LJ3.${i}_${j}.rst7 $i/$j/
    done
done
