#!/bin/sh
mkdir -p results/
max=8

max=$((4*max))

number_lines_rrt_point=0
number_lines_rrt_circle=0
number_lines_rrt_rectangle=0
number_lines_rrtstar_point=0
number_lines_rrtstar_circle=0
number_lines_rrtstar_rectangle=0




i=0
FILE="results/rrt-point.txt"

if [ -f $FILE ];
	then
	number_lines_rrt_point=`wc --lines < $FILE`
fi


while [ $number_lines_rrt_point -le $max ] && [ $i -le $((max/4-number_lines_rrt_point/4)) ]
do
	gnome-terminal --tab --title="tab-rrt-point-${i}" --command="python3 code/assign1-RRT_star.py --save --blind";
	if [ -f $FILE ];
	then
	number_lines_rrt_point=`wc --lines < $FILE`
	fi
	i=$((i+1))
done

i=0
FILE="results/rrt-circle.txt"


if [ -f $FILE ];
	then
	number_lines_rrt_circle=`wc --lines < $FILE`
fi

while [ $number_lines_rrt_circle -le $max ] && [ $i -le $((max/4-number_lines_rrt_circle/4)) ]
do
	gnome-terminal --tab --title="tab-rrt-circle-${i}" --command="python3 code/assign1-RRT_star.py --geom circle --save --blind";
	if [ -f $FILE ];
	then
	number_lines_rrt_circle=`wc --lines < $FILE`
	fi
	i=$((i+1))
done

i=0
FILE="results/rrt-rectangle.txt"

if [ -f $FILE ];
	then
	number_lines_rrt_rectangle=`wc --lines < $FILE`
fi


while [ $number_lines_rrt_rectangle -le $max ] && [ $i -le $((max/4-number_lines_rrt_rectangle/4)) ]
do
	gnome-terminal --tab --title="tab-rrt-rectangle-${i}" --command="python3 code/assign1-RRT_star.py --geom rectangle --save --blind";
	if [ -f $FILE ];
	then
	number_lines_rrt_rectangle=`wc --lines < $FILE`
	fi
	i=$((i+1))
done


i=0
FILE="results/rrtstar-point.txt"

if [ -f $FILE ];
	then
	number_lines_rrtstar_point=`wc --lines < $FILE`
fi


while [ $number_lines_rrtstar_point -le $max ] && [ $i -le $((max/4-number_lines_rrtstar_point/4)) ]
do
	gnome-terminal --tab --title="tab-rrtstar-point-${i}" --command="python3 code/assign1-RRT_star.py --alg rrtstar --save --blind";
	if [ -f $FILE ];
	then
	number_lines_rrtstar_point=`wc --lines < $FILE`
	fi
	i=$((i+1))
done
i=0
FILE="results/rrtstar-circle.txt"


if [ -f $FILE ];
	then
	number_lines_rrtstar_circle=`wc --lines < $FILE`
fi

while [ $number_lines_rrtstar_circle -le $max ] && [ $i -le $((max/4-number_lines_rrtstar_circle/4)) ]
do
	gnome-terminal --tab --title="tab-rrtstar-circle-${i}" --command="python3 code/assign1-RRT_star.py --alg rrtstar --geom circle --save --blind";
	if [ -f $FILE ];
	then
	number_lines_rrtstar_circle=`wc --lines < $FILE`
	fi
	i=$((i+1))
done
i=0
FILE="results/rrtstar-rectangle.txt"


if [ -f $FILE ];
	then
	number_lines_rrtstar_rectangle=`wc --lines < $FILE`
fi


while [ $number_lines_rrtstar_rectangle -le $max ] && [ $i -le $((max/4-number_lines_rrtstar_rectangle/4)) ]
do
	gnome-terminal --tab --title="tab-rrtstar-rectangle-${i}" --command="python3 code/assign1-RRT_star.py --alg rrtstar --geom rectangle --save --blind";
	if [ -f $FILE ];
	then
	number_lines_rrtstar_rectangle=`wc --lines < $FILE`
	fi
	i=$((i+1))
done