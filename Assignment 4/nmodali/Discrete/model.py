import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma,device

import pyximport; pyximport.install()

class VanillaNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(VanillaNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc_1 = nn.Linear(num_inputs, 16)
        self.fc_2 = nn.Linear(16,16)
        self.fc_3 = nn.Linear(16, num_outputs)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight,0.01)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        policy = F.softmax(self.fc_3(x),dim=-1)
        return policy

    @classmethod
    def train_model(cls, net, array_of_transitions, optimizer,args):
        loss=0;
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
                        advantage_rewards.append(running_return.cpu())
                advantage_mean = np.mean(advantage_rewards)
                advantage_stddev = np.std(advantage_rewards)
        for e in range(len(array_of_transitions)):
            transitions = array_of_transitions[e]
            states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

            states = torch.stack(states).to(device)
            actions = torch.stack(actions).to(device)
            rewards = torch.Tensor(rewards).to(device)
            masks = torch.Tensor(masks).to(device)

            returns = torch.zeros_like(rewards)

            cumm_reward = 0
            running_return = 0
            if args.alg_type == "curr-rwd":
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + gamma * running_return * masks[t]
                    cumm_reward = cumm_reward + rewards[t]
                    returns[t] = running_return
            elif args.alg_type == "full-rwd":
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + gamma * running_return * masks[t]
                    cumm_reward = cumm_reward + rewards[t]
                for t in reversed(range(len(rewards))):
                    returns[t] = running_return
            elif args.alg_type == "adv":
                for t in reversed(range(len(rewards))):
                    cumm_reward = cumm_reward + rewards[t]
                    running_return = (rewards[t] + gamma * running_return * masks[t])
                    returns[t] = (running_return - advantage_mean)/advantage_stddev

            avg_reward = avg_reward + cumm_reward.cpu()
            policies = net(states)
            policies = policies.view(-1, net.num_outputs)

            log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)

            loss = loss + (-log_policies * returns.detach()).sum()
        loss= loss/len(array_of_transitions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.detach(), avg_reward/len(array_of_transitions)

    def get_action(self, input,device=device):
        policy = self.forward(input)
        policy = policy.cpu()[0].data.numpy()

        action = np.random.choice(self.num_outputs, None, p=policy)
        return action