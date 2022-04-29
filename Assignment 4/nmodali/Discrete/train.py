import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import VanillaNet
from tensorboardX import SummaryWriter

from replaybuffer import ReplayBuffer
from config import env_name, log_interval, device, lr, gamma, env_seed, torch_seed
import matplotlib.pyplot as plt

import pyximport; pyximport.install()

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


def main():
    args = get_args()
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = VanillaNet(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    net.to(device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0
    avg_rewards=[]
    result_iterations=[]

    plt.figure()

    for iteration in range(args.num_iterations):
        replaybuffers=[]
        for e in range(args.num_episodes):
            done = False
            replaybuffer = ReplayBuffer()
            score = 0
            state = env.reset()
            state = torch.Tensor(state).to(device)
            state = state.unsqueeze(0)

            while not done:
                steps += 1

                action = net.get_action(state)
                next_state, reward, done, _ = env.step(action)

                next_state = torch.Tensor(next_state)
                next_state = next_state.unsqueeze(0).to(device)

                mask = 0 if done else 1
                reward = reward if not done else 0

                action_one_hot = torch.zeros(2)
                action_one_hot[action] = 1
                replaybuffer.push(state, next_state, action_one_hot, reward, mask)

                state = next_state
            replaybuffers.append(replaybuffer.sample())
        loss,avg_reward = VanillaNet.train_model(net, replaybuffers, optimizer,args) #

        avg_rewards.append(avg_reward)
        result_iterations.append(iteration)


        plt.cla()
        plt.plot(result_iterations,avg_rewards)
        if args.show_plot:
            plt.pause(0.05)

        if iteration % log_interval == 0:
            print('{} iteration | avg_reward: {:.2f}'.format(
                iteration, avg_reward))
            writer.add_scalar('log/score', float(avg_reward), iteration)
            writer.add_scalar('log/loss', float(loss), iteration)


    model_path='{}_ep_{}_it_{}_type_{}.pkl'.format(args.rl_alg,args.num_episodes,args.num_iterations,args.alg_type)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    save_state(net, optimizer, torch_seed, env_seed, os.path.join(args.model_path,model_path))
    with open("{}_details.txt".format(os.path.join(args.model_path,model_path[:-4])),"w") as f:
        f.write("env_name = {}\n".format(env_name))
        f.write("gamma = {}\n".format(gamma))
        f.write("lr = {}\n".format(lr))
        f.write("env_seed = {}\n".format(env_seed))
        f.write("torch_seed = {}\n".format(torch_seed))


    plt.title("Average Cumm Reward vs iteration")
    plt.xlabel("iteration")
    plt.ylabel("Average Reward")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    plt.savefig("{}/{}_plot.png".format(args.save_path,model_path[:-4]))

if __name__=="__main__":
    main()