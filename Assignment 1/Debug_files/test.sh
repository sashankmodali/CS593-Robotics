#!/bin/sh
mkdir -p results/
max=8

number_lines_rrt_point=0
number_lines_rrt_circle=0
number_lines_rrt_rectangle=0
number_lines_rrtstar_point=0
number_lines_rrtstar_circle=0
number_lines_rrtstar_rectangle=0



# number_lines_rrt_point=`wc --lines < "results-rrt-point.txt"`
i=0
FILE="results-rrt-circle.txt"
while [ $number_lines_rrt_circle -le $max ] && [ $i -le $((max/2)) ]
do
	if [ -f $FILE ];
	then
	number_lines_rrt_circle=`wc --lines < $FILE`
	fi
	echo $number_lines_rrt_circle
	i=$((i+1))
done