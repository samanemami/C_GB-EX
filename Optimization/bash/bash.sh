#!/bin/bash

line="----------"
n=0

echo $line
echo "starting at: $(date)"; echo $line

for lr in 0.025 0.05 0.1 0.5 1
do
    for dp in 2 5 10 20
    do
        echo "iter:" $n; echo $line
        sleep $n
        ((n=n+1))

        python3 del.py $lr $dp train1
        IFS=','
        while read -r score depth learning_rate
        do
            score1=$score
        done < results.csv

        python3 del.py $lr $dp train2
        IFS=','
        while read -r score depth learning_rate
        do
            score=`echo "$score1 + $score" | bc`
            score=$(awk "BEGIN {print $score/2}")
            printf '%s,%s,%s\n' "$score" "$depth" "$learning_rate" >> mean_test_score.csv
        done < results.csv
    done
done

python3 del.py 1 1 test1

echo $line
echo "UP Time"; uptime; echo $line
echo "memory"; free; echo $line
echo "Finishing at $(date)"; echo $line
