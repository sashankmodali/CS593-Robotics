'''Define a simple 2d-arm environment
'''
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np
from modified_gym_env.reacher_fixed_goal import Reacher


class ReacherBulletEnv(BaseBulletEnv):
    '''
    A wrapper for the reacher environment to be used with the 
    Pybulletgym package.
    '''
    def __init__(self, rand_init=True):
        '''
        Initialize the robot.
        :param rand_init: If True, randomly initialize the robot starting position.
        '''
        self.robot = Reacher(rand_init)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        '''
        Define the scence of the robot.
        :param bullet_client: 
        :returns pybulletgym.envs.roboschool.scences.scene_bases.SingleRobotEmptyScene object
        '''
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        '''
        Take single step.
        :param a: The action to take for this current step.
        :returns tuple: Returns a tuple consisting of (next state, reward, termination condition,{})
        '''
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()
        done = False

        state = self.robot.calc_state()  # sets self.to_target_vec

        self.potential = self.robot.calc_potential()*0.01

        electricity_cost = (
             - 0.1 * (a[0]**2 + a[1]**2)  # stall torque require some energy
        )
        self.rewards = [float(self.potential), float(electricity_cost)]
        if abs(self.potential) <0.01:
            done = True
            self.rewards = [0]
        self.HUD(state, a, False)
        return state, sum(self.rewards) , done, {}

    def camera_adjust(self):
        '''
        A method to set the camera to a particular position.
        '''
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)