#!/bin/bash

python mpnet_test.py --data-path "./data/dataset-c3d/" --env-type r3d --s 0 --sp 2000 --N 3 --vis-dir "./plots/" --part "compare"
python mpnet_test.py --data-path "./data/dataset-c3d/" --env-type r3d --n-runs 5  --s 0 --sp 2000 --vis-dir "./plots/" --part "runs"