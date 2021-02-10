
for i in $(seq 50.0 0.1 60.0)
do
    echo $i
    mkdir $i
    sed -e "s|XXX|$i|g" LJ2.leap > test.leap
    tleap -f test.leap
    mv LJ2.$i.rst7 $i/
done
