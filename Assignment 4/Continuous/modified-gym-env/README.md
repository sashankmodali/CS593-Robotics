# A 2-link arm Reach task.

## Install

Clone the package to your system: 

```
git clone https://github.com/ucsdarclab/pybullet-gym-env.git
```

install it using pip:

```
pip install -e pybullet-gym-env/
```


## Usage

Once the package is installed, you can create an environment using the following command:

```
env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
```

If `rand_init=True`, then the arm will initialize to a random location after each reset.