import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy

import os

from signal import signal, SIGINT

import pyximport; pyximport.install()

from tensorboardX import SummaryWriter

from config import env_name, log_interval, device, lr, gamma, env_seed, torch_seed, rand_init, model_variance


# class SigmaNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(SigmaNet, self).__init__()
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs

#         self.fc_1 = nn.Linear(num_inputs, 128)
#         self.fc_2 = nn.Linear(128, num_outputs)

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight,mode='fan_in', nonlinearity='relu')

#     def forward(self, input):
#         x = F.leaky_relu(self.fc_1(input))
#         x = torch.exp(self.fc_2(x))
#         sigma = (torch.diag_embed(torch.add(0*torch.flatten(x,start_dim = -2,end_dim=-1),0.1)))
#         # print("x shape : {} , sigma shape : {}".format(x.shape,sigma.shape))
#         return sigma

class MeanNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MeanNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


        # self.fc_1 = nn.Linear(num_inputs, 128)    # for 128 1 layer network
        # self.fc_2 = nn.Linear(128, num_outputs)   # for 128 1 layer network

        self.fc_1 = nn.Linear(num_inputs,32)   # for 32 2 layer network
        self.fc_2 = nn.Linear(32,32)            # for 32 2 layer network
        self.fc_5 = nn.Linear(32, num_outputs)  # for 32 2 layer network

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_in', nonlinearity='relu')

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        x = F.relu(self.fc_2(x))
        mean = torch.flatten(F.relu(self.fc_5(x)),start_dim=-2)
        # print("mean shape {}".format(mean.shape))
        return mean


class GaussianNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(GaussianNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.mean_net = MeanNet(num_inputs,num_outputs)
        # self.sigma_net = torch.diag_embed(torch.ones(num_outputs)*model_variance) # First Neural Net was used, but this was finalised. 
        # self.sigma_net = SigmaNet(num_inputs,num_outputs) # # for 128 1 layer network

    def forward(self, input):
        mean_tensor = self.mean_net(input).to(device) 

        return mean_tensor

    @classmethod
    def train_model(cls, net, array_of_transitions, optimizer,args,plt):
        def save_state(net, opt, torch_seed, env_seed, fname, np_seed=None, py_seed=None,):
            # save both model state and optimizer state
            states = {
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
                'torch_seed': torch_seed,
                'env_seed': env_seed,
                'gamma': gamma,
                'lr' : lr
            }
            torch.save(states, fname)


        def handler(signal_received, frame):
            # Handle any cleanup here
            print('SIGINT or CTRL-C detected. Exiting gracefully')
            if type(args.load_train) == list and (args.load_train[0] >-1 and args.load_train[1] >-1) :
                model_path='{}_ep_{}_it_{}_type_{}_interrupted.pkl'.format(args.rl_alg,args.num_episodes,args.load_train[1]+args.num_iterations,args.alg_type)
            else:    
                model_path='{}_ep_{}_it_{}_type_{}_interrupted.pkl'.format(args.rl_alg,args.num_episodes,args.num_iterations,args.alg_type)
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            save_state(net, optimizer, torch_seed, env_seed, os.path.join(args.model_path,model_path))

            plt.title("Average Cumm Reward vs iteration")
            plt.xlabel("iteration")
            plt.ylabel("Average Reward")
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            plt.savefig("{}/{}_plot.png".format(args.save_path,model_path[:-4]))

            exit(0)

        signal(SIGINT, handler)
        avg_reward=0;
        with torch.no_grad():
            if args.alg_type == "adv":
                advantage_rewards=[];
                for e in range(len(array_of_transitions)):
                    transitions = array_of_transitions[e]
                    states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

                    states = torch.stack(states).to(device)
                    actions = torch.stack(actions).to(device)
                    rewards = torch.Tensor(rewards).to(device)
                    masks = torch.Tensor(masks).to(device)
                    returns = torch.zeros_like(rewards)
                    running_return = 0
                    for t in reversed(range(len(rewards))):
                        running_return = rewards[t] + gamma * running_return * masks[t]
                        advantage_rewards.append((running_return.cpu()))

                advantage_mean = np.mean(advantage_rewards)
                advantage_stddev = np.std(advantage_rewards)
        loss=0
        for e in range(len(array_of_transitions)):
            transitions = array_of_transitions[e]
            states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            rewards = torch.Tensor(rewards).to(device)
            masks = torch.Tensor(masks).to(device)

            returns = torch.zeros_like(rewards).to(device)
            log_probs = torch.zeros_like(rewards).to(device)

            mean_tensor = net(states)
            sigma_tensor = torch.diag_embed(torch.ones(net.num_outputs)*model_variance)
            sigma_tensor = sigma_tensor.repeat(mean_tensor.shape[0],1,1)
            sigma_tensor = sigma_tensor.to(device)
            policies = [torch.distributions.multivariate_normal.MultivariateNormal(mean_tensor[i], sigma_tensor[i]) for i in range(len(mean_tensor))]

            cumm_reward = 0
            running_return = 0
            if args.alg_type == "curr-rwd":
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + gamma * running_return * masks[t]
                    cumm_reward = cumm_reward + rewards[t]
                    returns[t] = running_return
                    log_probs[t] = policies[t].log_prob(actions[t])
            elif args.alg_type == "full-rwd":
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + gamma * running_return * masks[t]
                    cumm_reward = cumm_reward + rewards[t]
                    log_probs[t] = policies[t].log_prob(actions[t])
                for t in reversed(range(len(rewards))):
                    returns[t] = running_return
            elif args.alg_type == "adv":
                for t in reversed(range(len(rewards))):
                    cumm_reward = cumm_reward + rewards[t]
                    running_return = (rewards[t] + gamma * running_return * masks[t])
                    if advantage_stddev>0.01:
                        returns[t] = (running_return - advantage_mean)/advantage_stddev
                    else:
                        returns[t] = 0
                    log_probs[t] = policies[t].log_prob(actions[t])

            avg_reward = avg_reward + cumm_reward.detach().cpu()
            loss= loss + (-log_probs* returns).sum()
        #     print("logprobs :{}".format(log_probs.shape))
        #     print("returns :{}".format(returns.shape))
        # print("advantage_stddev :{}".format(advantage_stddev))
        # print("advantage_mean :{}".format(advantage_mean))
            
        optimizer.zero_grad()
        loss= loss/len(array_of_transitions)
        loss.backward()
        optimizer.step()
        avg_reward=avg_reward/len(array_of_transitions)
        
        return loss.detach().numpy(), avg_reward

    def get_action(self, input,device=device):
        mean_tensor = self.forward(input)
        sigma_tensor = torch.diag_embed(torch.ones(self.num_outputs)*model_variance)
        sigma_tensor = sigma_tensor.to(device)
        policy = torch.distributions.multivariate_normal.MultivariateNormal(mean_tensor, sigma_tensor)

        action = policy.sample().cpu()
        # print("action is : {}".format(action))
        return action.detach()