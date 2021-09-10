#/bin/bash
export CUDA_VISIBLE_DEVICES=0

for fp in $(seq 3.5 0.5 16.0)
do
    sed -e "s/XXX/$fp/g" cpptraj.in > test.cpptraj
    cpptraj -i test.cpptraj
done


