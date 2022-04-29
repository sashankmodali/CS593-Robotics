# A 2-link arm Reach task. (borrowed from https://github.com/ucsdarclab/pybullet-gym-env.git)

## Install

1. Clone the package to your system.

2. install it using pip:

	```
	pip install -e pybullet-gym-env/
	```


## Usage

Once the package is installed, you can create an environment using the following command:

```
env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
```

If `rand_init=True`, then the arm will initialize to a random location after each reset.