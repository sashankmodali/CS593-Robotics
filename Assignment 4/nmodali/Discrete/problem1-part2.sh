#!/bin/bash

tab="--tab"
cmd="bash -c 'java RunRTSPClient';bash"
foo=""

foo+=($tab -e "python train.py --num-episodes 1000 --num-iterations 200 --alg-type adv")
foo+=($tab -e "python train.py --num-episodes 300 --num-iterations 200 --alg-type adv")
foo+=($tab -e "python train.py --num-episodes 100 --num-iterations 200 --alg-type adv")


gnome-terminal "${foo[@]}"

exit 0