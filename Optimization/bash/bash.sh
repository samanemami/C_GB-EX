#!/bin/bash

line="----------------"

echo $line
echo "starting at: $(date)"; echo $line


for lr in 0.025 0.05 0.1 0.5 1
do
    for dp in 2 5 10 20
    do
        python3 opt.py $lr $dp train1 1
        python3 opt.py $lr $dp train2 1
    done
done

echo "starting at: $(date)"; echo $line
echo "UP Time"; uptime; echo $line
echo "memory"; free; echo $line
echo "Finishing at $(date)"; echo $line
