"""Register new 2 Link robot arm"""
from gym.envs.registration import register

register(
    id='ReacherPyBulletEnv-v1',
    entry_point='modified_gym_env.reacher_env_mod:ReacherBulletEnv',
    kwargs={'rand_init':True},
    max_episode_steps=150,
    reward_threshold=18.0,
)
