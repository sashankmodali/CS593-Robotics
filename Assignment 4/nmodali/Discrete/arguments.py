import argparse
import math
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Vanilla Reinforce')

    ## General Arguments
    parser.add_argument('--rl-alg', type=str, default="vanilla",help="Name of RL algorithm")
    parser.add_argument('--alg-type', type=str, default="curr-rwd",help="Update target type")
    parser.add_argument('--num-episodes', type=int, default=1000, help="Number of episodes for training")
    parser.add_argument('--num-iterations', type=int, default=30, help="Number of iterations for training")
    parser.add_argument('--model-path', type=str, default="./pretrained_models",help="Directory to save trained models")
    parser.add_argument('--save-path', type=str, default="./results",help="Directory to save plots")
    parser.add_argument('--show-plot', default=False, action="store_true",help=" Boolean to visualize average reward plot")

    args = parser.parse_args()

    return args
