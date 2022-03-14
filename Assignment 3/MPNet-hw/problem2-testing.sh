#!/bin/bash

tab="--tab"
cmd="bash -c 'java RunRTSPClient';bash"
foo=""
foo+=($tab -e "python mpnet_test.py --data-path \"./data/dataset-c3d/\" --epoch 2000 --env-type r3d --s 0 --sp 2000 --N 10 --NP 100 --vis-dir \"\" --part \"all\"")
foo+=($tab -e "python mpnet_test.py --data-path \"./data/dataset-c3d/\" --epoch 2000 --env-type r3d --s 0 --sp 2000 --N 10 --NP 100 --vis-dir \"\" --part \"all\" --dropout-disabled")
foo+=($tab -e "python mpnet_test.py --data-path \"./data/dataset-c3d/\" --epoch 2000 --env-type r3d --s 0 --sp 2000 --N 10 --NP 100 --vis-dir \"\" --part \"all\" --lvc-disabled")

gnome-terminal "${foo[@]}"

exit 0