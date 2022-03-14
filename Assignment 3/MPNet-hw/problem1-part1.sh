#!/bin/bash

tab="--tab"
cmd="bash -c 'java RunRTSPClient';bash"
foo=""
foo+=($tab -e "python mpnet_test.py --data-path \"./data/dataset-s2d/\" --N 100 --NP 100 --part all --vis-dir \"\" ")
foo+=($tab -e "python mpnet_test.py --data-path \"./data/dataset-s2d/\" --N 100 --NP 100 --part all --vis-dir \"\" --dropout-disabled")
foo+=($tab -e "python mpnet_test.py --data-path \"./data/dataset-s2d/\" --N 100 --NP 100 --part all --vis-dir \"\" --lvc-disabled")

gnome-terminal "${foo[@]}"

exit 0




