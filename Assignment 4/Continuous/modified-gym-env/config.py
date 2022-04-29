import torch

env_name = 'modified_gym_env:ReacherPyBulletEnv-v1'
rand_init = True
gamma = 0.9
lr = 1e-2
log_interval = 10
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
env_seed = 500
torch_seed = 500
model_variance = 0.1