#!/bin/bash

line="----------"
n=0
fold1=train1
fold2=train2

echo $line
echo "starting at: $(date)"; echo $line


for lr in 0.025 0.05 0.1 0.5 1
do
    for dp in 2 5 10 20
    do
        echo "iter:" $n; echo $line
        sleep $n
        ((n=n+1))
        echo "`ps aux --sort -rss`"; echo $line

        python3 del.py $lr $dp train1 1
        
        INPUT=reslt.csv
        OLDIFS=$IFS
        IFS=','
        echo "first fold"
        while read -r score depth learning_rate
        do
            echo "score: $score"
            echo "depth : $depth"
            echo "learning_rate : $learning_rate"
        done < $INPUT
        IFS=$OLDIFS
        
        echo $line

        python3 del.py $lr $dp train2 1 
        echo "second fold"
        while read -r score depth learning_rate
        do
            echo "score: $score"
            echo "depth : $depth"
            echo "learning_rate : $learning_rate"
        done < $INPUT
        IFS=$OLDIFS
        
    done
done


echo $line
echo "UP Time"; uptime; echo $line
echo "memory"; free; echo $line
echo "Finishing at $(date)"; echo $line
