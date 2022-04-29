import argparse
from dis import dis
import gym
import numpy as np
from itertools import count


import pyximport; pyximport.install()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import pybullet
import os

from tensorboardX import SummaryWriter

from config import env_name, log_interval, device, lr, gamma, env_seed, torch_seed, rand_init

from arguments import get_args


def save_state(net, opt, torch_seed, env_seed, fname, np_seed=None, py_seed=None,):
    # save both model state and optimizer state
    states = {
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'torch_seed': torch_seed,
        'env_seed': env_seed,
        'gamma': gamma,
        'lr' : lr
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
    state = torch.from_numpy(state).float().to(device)
    mu = net(state).to(device)
    #calculate the probability
    dist = MultivariateNormal(mu, 0.1*torch.eye(2).to(device))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob
 
def update_policy(policy_loss,optimizer, device):
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum().to(device) 
    policy_loss.backward()
    optimizer.step()

def main():
    args = get_args()
    gym.logger.set_level(40)
    env = gym.make(env_name,rand_init=rand_init)
    env.seed(env_seed)
    torch.manual_seed(torch_seed)

    state= env.reset()

    num_inputs = state.shape[0]
    num_actions = env.action_space.shape[0]
    print('state size:', num_inputs)
    print('action space shape, highs, lows : ', env.action_space.shape," ",env.action_space.high," ",env.action_space.low)


    net = GaussNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    writer = SummaryWriter('logs')


    if type(args.load_train) == list and (args.load_train[0] >-1 and args.load_train[1] >-1) :
        load_path='{}_ep_{}_it_{}_type_{}.pkl'.format(args.rl_alg,args.load_train[0],args.load_train[1],args.alg_type)
        if not os.path.exists(os.path.join(args.model_path,load_path)):
            raise Exception("load path {} doesn't exist".format(load_path))
        os.system('clear')
        print(" Loading ... {}".format(load_path))

        net.load_state_dict(torch.load(os.path.join(args.model_path,load_path))["state_dict"])
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path,load_path))["optimizer"])
        start_iteration = args.load_train[1]
        start_episode = 0
    else:
        start_iteration = 0
        start_episode = 0
    net.train()

    steps=0
    
    avg_itr_reward = []
    num_iter = args.num_iterations
    num_episodes = args.num_episodes
    for itr in range(start_iteration,start_iteration + num_iter):
        total_ep_reward = 0
        log_probs_itr = []
        policy_loss_itr = []
        returns_itr = []
        for i_episode in range(num_episodes):
            state, ep_reward = env.reset(), 0
            done = False
            rewards = []
            log_probs = []
            returns=[]
            while not done:
                steps=steps+1;
                action, log_prob = get_action(state, net, device)
                state, reward, done, _ = env.step(action.cpu().numpy())
                # print(reward)
                ep_reward +=reward
                rewards.append(reward)  
                log_probs.append(log_prob)     
            #Compute the returns from the rewards 
            R = 0
            for r in rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(device)
            #Store the log probability and returns to calculate the loss function
            log_probs_itr.append(log_probs)
            returns_itr.append(returns)
            total_ep_reward += ep_reward     
        
        returns_ep = torch.cat(returns_itr)
        # Find the mean and std of the returns over the episodes to compute 
        mean  = returns_ep.mean()
        std = returns_ep.std()
        for log_probs, returns in zip(log_probs_itr,returns_itr):
            # Use VANILLA REINFORCE with baseline
            returns = (returns - mean) / std
            policy_loss_ep = []
            for log_prob, return_ in zip(log_probs,returns):
                policy_loss_ep.append(-log_prob * return_)
            policy_loss_ep = torch.stack(policy_loss_ep).to(device)
            policy_loss_itr.append(policy_loss_ep)        
        avg_itr_reward.append(total_ep_reward / num_episodes)  
        # Update the policy neural network parameter using the Loss function 
        update_policy(policy_loss_itr,optimizer,device)
        if itr % log_interval == 0:
            print('iteration: {} | loss: {:.3f} | steps: {} | avg_reward: {:.2f}'.format(
                itr, policy_loss_ep.sum(), steps, avg_itr_reward[-1]))
            writer.add_scalar('log/avg_reward', float(avg_itr_reward[-1]), itr)
            writer.add_scalar('log/loss', float(policy_loss_ep.sum()), itr)
    if type(args.load_train) == list and (args.load_train[0] >-1 and args.load_train[1] >-1) :
        model_path='{}_ep_{}_it_{}_type_{}.pkl'.format(args.rl_alg,args.num_episodes,args.load_train[1]+args.num_iterations,args.alg_type)
    else:    
        model_path='{}_ep_{}_it_{}_type_{}.pkl'.format(args.rl_alg,args.num_episodes,args.num_iterations,args.alg_type)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    save_state(net, optimizer, torch_seed, env_seed, os.path.join(args.model_path,model_path))

    plt.figure()
    plt.plot(range(itr+1),avg_itr_reward)
    plt.title("Average Cumulative Reward vs iteration")
    plt.xlabel("iteration")
    plt.ylabel("Average Reward")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    plt.savefig("{}/{}_plot.png".format(args.save_path,model_path[:-4]))
    
if __name__ == '__main__':
    main()