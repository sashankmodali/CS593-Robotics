'''Define a wrapper for reading the mujoco 2D link arm description.
'''
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import numpy as np


class Reacher(MJCFBasedRobot):
    '''
    An object to load reacher.xml as a pybullet object.
    '''
    RADIUS_LIMIT = 0.21

    def __init__(self, rand_init=False):
        '''
        Initialize the robot
        :param rand_init: If True, randomly initialize robot position.
        '''
        MJCFBasedRobot.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=9)
        self.rand_init = rand_init

    def robot_specific_reset(self, bullet_client):
        '''
        Reset the robot.
        :param bullet_client: The client object.
        '''
        r = 0.15
        theta = np.pi/4
        self.jdict["target_x"].reset_current_position(r * np.cos(theta), 0)
        self.jdict["target_y"].reset_current_position(r * np.sin(theta), 0)
        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint = self.jdict["joint1"]
        # self.central_joint.reset_current_position(self.np_random.uniform(low=-np.pi/2, high=np.pi/2), 0)
        self.central_joint.reset_current_position(np.pi/2, 0)
        if self.rand_init:
            self.elbow_joint.reset_current_position(self.np_random.uniform(low=-np.pi/2, high=np.pi/2), 0)
        else:
            self.elbow_joint.reset_current_position(np.pi/4, 0)

    def apply_action(self, a):
        '''
        Apply the action a to the robot.
        :param a: A numpy array, of the current action.
        '''
        assert (np.isfinite(a).all())
        self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        '''
        Update the robot state.
        '''
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x/0.27,
            target_y/0.27,
            self.to_target_vec[0]/0.27,
            self.to_target_vec[1]/0.27,
            # np.cos(theta),
            # np.sin(theta),
            theta/np.pi,
            self.theta_dot,
            self.gamma/np.pi,
            self.gamma_dot,
        ])

    def calc_potential(self):
        '''
        Calculate the distance toward goal.
        '''
        return -100 * np.linalg.norm(self.to_target_vec)
