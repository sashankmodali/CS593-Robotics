import os
import sys
import gym
import random
import numpy as np

import pybullet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal

from tensorboardX import SummaryWriter

from replaybuffer import ReplayBuffer

import matplotlib.pyplot as plt
import pyautogui
import cv2

import copy

from signal import signal, SIGINT
from sys import exit

from config import env_name, log_interval, device, lr, gamma, env_seed, torch_seed, rand_init

from arguments import get_args

def save_state(net, opt, torch_seed, env_seed, fname, np_seed=None, py_seed=None,):
    # save both model state and optimizer state
    states = {
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'torch_seed': torch_seed,
        'env_seed': env_seed
        # 'py_seed': py_seed,
        # 'np_seed': np_seed
    }
    torch.save(states, fname)

class GaussNet(nn.Module):
    def __init__(self):
        super(GaussNet, self).__init__()
        self.affine1 = nn.Linear(8, 32)
        self.linear = nn.Linear(32,32)
        self.affine2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.linear(x)
        x = F.relu(x)
        mu = self.affine2(x)
        x = F.relu(x)
        return mu

def get_action(state, net, device):
    mu = net(state).to(device)
    #calculate the probability
    dist = MultivariateNormal(mu, 0.1*torch.eye(2).to(device))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.detach(), log_prob.detach()

def main():
    args = get_args()
    gym.logger.set_level(40)
    env = gym.make(env_name,rand_init=rand_init)
    env.seed(env_seed)
    # torch.manual_seed(torch_seed)

    env.render(mode="human")

    state=env.reset()
    num_inputs = state.shape[0]
    num_actions = env.action_space.shape[0]
    print('state size:', env.observation_space.shape)
    print('action space shape, highs, lows : ', env.action_space.shape," ",env.action_space.high," ",env.action_space.low)

    net = GaussNet()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    net.to(device)


    model_path='{}_ep_{}_it_{}_type_{}.pkl'.format(args.rl_alg,args.num_episodes,args.num_iterations,args.alg_type)
    if not os.path.exists(args.model_path):
        raise Exception("Model Path doesn't exist")
    # save_state(net, optimizer, torch_seed, env_seed, os.path.join(args.model_path,model_path))    
    print(" Loading ... {}".format(model_path))
    net.load_state_dict(torch.load(os.path.join(args.model_path,model_path))["state_dict"])
    optimizer.load_state_dict(torch.load(os.path.join(args.model_path,model_path))["optimizer"])



    net.eval()
    running_score = 0
    steps = 0
    loss = 0
    avg_rewards=[]
    result_iterations=[]

    SCREEN_SIZE = tuple(pyautogui.size())
    # define the codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # frames per second
    fps = 25.0
    # create the video write object
    out = cv2.VideoWriter("{}/{}_video.avi".format(args.save_path,model_path[:-4]), fourcc, fps, (SCREEN_SIZE))
    record_seconds = 15

    plt.figure()

    def handler(signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        exit(0)

    for iteration in range(args.num_iterations):
        if iteration == 0:
            signal(SIGINT, handler)
            
        env.render(mode="human")
        done = False
        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:

             # make a screenshot
            img = pyautogui.screenshot()
            # convert these pixels to a proper numpy array to work with OpenCV
            frame = np.array(img)
            # convert colors from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # write the frame
            out.write(frame)
            # show the frame
            # cv2.imshow("screenshot", frame)
            # if the user clicks q, it exits
            if cv2.waitKey(1) == ord("q"):
                break
            steps += 1
            action, log_prob = get_action(state,net,device)
            # print(action.numpy().flatten())
            next_state, reward, done, _ = env.step(action.numpy().flatten())

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0).to(device)

            mask = 0 if done else 1
            reward = reward if not done else 0

            score = score + reward

            state = next_state

        running_score = running_score + score

        if iteration % log_interval == 0:
            print('{} iteration | total_reward: {:.2f}'.format(
                iteration, score))

if __name__=="__main__":
    main()