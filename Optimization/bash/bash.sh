#!/bin/bash

module load python/#version

for lr in {0.025, 0.05, 0.1, 0.5, 1}
do
    for dp in {2, 5, 10, 20}
    do
        python opt.py $lr $dp train1 seed
        python opt.py $lr $dp train2
    done
done
