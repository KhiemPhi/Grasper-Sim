import os
import random
import sys
from configparser import ConfigParser
from math import pi, sqrt
from time import sleep, time

import astropy.coordinates
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance
from transforms3d import euler


import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 20*100

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def step(self, action):
        '''
        Now we are ready to determine what will happen with each env.step(action) command. 
        As I mentioned before the actions are the Cartesian position of the gripper plus a joint variable for both fingers. 
        
        I am going to use pybullet.calculateInverseKinematics() for calculating target joint variables for the robot. 
        
        However, we gradually move the robot toward the desired Cartesian position using a variable called dv for smoother inverse kinematics output. 
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) is also used for better rendering. 
        
        For the sake of simplicity, the gripper orientation is considered to be perpendicular to the ground. 
        
        I have converted the angles to Quaternion variables using pybullet.getQuaternionFromEuler(). 
        
        In each step, I read the current Cartesian position (pybullet.getLinkState()) of the gripper and add the small variation toward the target Cartesian position, 
        then calculate the joint variables for reaching to that new Cartesian position ( pybullet.calculateInverseKinematics()), 
        then I apply those joint variables using pybullet.setJointMotorControlArray() instead of pybullet.setJointMotorControl2() to be a one-liner! 
        
        After attempting to interact with the environment we run the environment for one time step (pybullet.stepSimulation().) 
        
        Then, I read the state of object, robot gripper and its fingers. I return the gripper and its fingers state as the observation 
        but pass the state of the object as a diagnostic information useful for debugging. It can sometimes be useful for learning.  
        '''

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7] # all seven joints    

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        # pybullet.setJointMotorControl2([objectUid],[jointIndex],[controller],[targetPosition]), we can generate it this way 

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])

        if state_object[2]>0.45: # grasp and pick it up to a certain height 
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        
        info = {'object_position': state_object}
        
        
        self.observation = state_robot + state_fingers # Whatever you want 

        
        
        
        
        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF('ObjectURDFs/cube/cube_small.urdf', basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=1.5,
                                                            yaw=171.60,
                                                            pitch=-20.80,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=10.0)
        (_, _, px, depth, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))
        
        rgb_array = rgb_array[:, :, :3]

        

        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()


env = PandaEnv()

env.reset()
while True: 
    env.render()
    p.stepSimulation()