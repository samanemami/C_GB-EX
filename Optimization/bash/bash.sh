#!/bin/bash

line="----------------"
n=0

echo $line
echo "starting at: $(date)"; echo $line


for lr in 0.025 0.05 0.1 0.5 1
do
    for dp in 2 5 10 20
    do
        echo "iter:" $n;
        sleep $n
        ((n=n+1))
        
        python3 opt.py $lr $dp train1 1
        python3 opt.py $lr $dp train2 1
        
        INPUT=iris.csv
        OLDIFS=$IFS
        IFS=','
        while read -r score depth learning_rate
        do
            echo "score : $score"
            echo "depth : $depth"
            echo "learning_rate : $learning_rate"
        done < $INPUT
        IFS=$OLDIFS
    done
done

echo "starting at: $(date)"; echo $line
echo "UP Time"; uptime; echo $line
echo "memory"; free; echo $line
echo "Finishing at $(date)"; echo $line
