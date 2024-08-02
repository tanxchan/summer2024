#!/bin/bash
echo 'hello world'
read -p 'Month?' m
read -p 'Start Day?' d1
read -p 'End Day?' d2
read -p 'Year?' y

for ((i = $d1; i<= $d2; i++))
do
day=$i
if [ $i -lt 10 ];
then
day=0$i
fi
curl https://data.darts.isas.jaxa.jp/pub/calet/cal-v1.1/CHD/level1.1/obs/20$y/CHD_$y$m$day.dat -o /Users/tshao/Documents/data/$m$day$y.dat
done
