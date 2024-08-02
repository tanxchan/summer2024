#!/bin/bash
echo 'hello world'
read -p 'Start Year?' y1
read -p 'End Year?' y2
for ((i=$y1; i<=$y2; i++)) do
        y=$i
        if [ $i -le 9 ]; then
                y=0$i
        fi
        for ((j=1; j<=12; j++)) do
                m=$j
                if [ $m -le 9 ]; then
                        m=0$j
                fi
                #echo $y$m
                curl https://data.darts.isas.jaxa.jp/pub/seda/sdom/txt/seda-sdom_flux_20$y$m.txt -o ../data/$y$m.txt || echo 'Ion data not found'
                #https://data.darts.isas.jaxa.jp/pub/seda/sdom/txt/seda-sdom_flux_200908.txt
done
done
