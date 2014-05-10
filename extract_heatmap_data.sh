#!/bin/bash

for i in eta*; do 
    a=$(echo $i | awk -F "_" '{printf "%s \t %s",  $2, $4}')
    b=$(cat $i/Diodicities.dat | awk '/2/ {print}')
    echo $a $b
done
