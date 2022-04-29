#!/bin/bash

tab="--tab"
cmd="bash -c 'java RunRTSPClient';bash"
foo=""
foo+=($tab -e "python train.py --num-episodes 500 --num-iterations 200 --alg-type full-rwd")
foo+=($tab -e "python train.py --num-episodes 500 --num-iterations 200 --alg-type curr-rwd")
# foo+=($tab -e "python train.py --num-episodes 500 --num-iterations 200 --alg-type adv")


gnome-terminal "${foo[@]}"

exit 0