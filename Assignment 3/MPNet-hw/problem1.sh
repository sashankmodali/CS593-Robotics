#!/bin/bash


python mpnet_test.py --data-path "./data/dataset-s2d/" --N 3 --vis-dir "./plots/" --part "compare" ;

python mpnet_test.py --data-path "./data/dataset-s2d/" --n-runs 5 --vis-dir "./plots/" --part "runs" ;

./problem1-part1.sh
