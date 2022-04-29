import torch

env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.001
log_interval = 10
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
env_seed = 500
torch_seed = 500