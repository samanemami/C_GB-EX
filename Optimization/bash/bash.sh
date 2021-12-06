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

        python3 del.py $lr $dp train1 1
        
        IFS=','
        while read -r score depth learning_rate
        do
            printf '%s,%s,%s\n' "$score" "$depth" "$learning_rate" >> file.csv
            sum=`echo "$score + $score" | bc`
            echo $sum
        done < results.csv


        python3 del.py $lr $dp train2 1

        IFS=','
        while read -r score depth learning_rate
        do
            printf '%s,%s,%s\n' "$score" "$depth" "$learning_rate" >> file.csv
        done < results.csv
    done
done


echo $line
echo "UP Time"; uptime; echo $line
echo "memory"; free; echo $line
echo "Finishing at $(date)"; echo $line
