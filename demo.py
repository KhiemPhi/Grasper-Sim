import argparse
import math
import os
import random
from time import sleep, time
from urllib import robotparser

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import urdf_models.models_data as md


def move_arm_and_rotate(pandaUid, coeff=-math.pi/4): 
    p.setJointMotorControl2(pandaUid, 2, 
                            p.POSITION_CONTROL,0) # can modify               
    p.setJointMotorControl2(pandaUid, 4, 
                p.POSITION_CONTROL,0) # can modify
    p.setJointMotorControl2(pandaUid, 6, 
                p.POSITION_CONTROL,coeff) # can modify
    p.setJointMotorControl2(pandaUid, 0, 
                            p.POSITION_CONTROL,0) # do not modify
    p.setJointMotorControl2(pandaUid, 1, 
                    p.POSITION_CONTROL,math.pi/4.) # do not modify 
    p.setJointMotorControl2(pandaUid, 3, 
                    p.POSITION_CONTROL,-math.pi/2.) # do not modify
    p.setJointMotorControl2(pandaUid, 5,  
                    p.POSITION_CONTROL,3*math.pi/4) # do not modify
    
    p.setJointMotorControl2(pandaUid, 9, 
                    p.POSITION_CONTROL, 0.08)
    p.setJointMotorControl2(pandaUid, 10, 
                    p.POSITION_CONTROL, 0.08)

def approach_object(pandaUid):
    p.setJointMotorControl2(pandaUid, 1, 
                            p.POSITION_CONTROL,math.pi/4.+.15)
    p.setJointMotorControl2(pandaUid, 3, 
                            p.POSITION_CONTROL,-math.pi/2.+.15)

def grip_object(pandaUid):
    p.setJointMotorControl2(pandaUid, 9, 
                            p.POSITION_CONTROL, 0.0, force = 200)
    p.setJointMotorControl2(pandaUid, 10, 
                            p.POSITION_CONTROL, 0.0, force = 200)

def return_object_to_original_robot_pose(pandaUid): 
    p.setJointMotorControl2(pandaUid, 1, 
                            p.POSITION_CONTROL,math.pi/4.-1)
    p.setJointMotorControl2(pandaUid, 3, 
                            p.POSITION_CONTROL,-math.pi/2.-1)

def reset_object(objectUid, models, object_name, state_object): 
    p.removeBody(objectUid)
    objectUid =  p.loadURDF(models[object_name], basePosition=state_object)
    return objectUid

def get_grasp_img():
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.52,-0.64,0.14],
                                                                distance=1.20,
                                                                yaw=168.0,
                                                                pitch=-11.20,
                                                                roll=0,
                                                                upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                aspect=float(960) /720,
                                                nearVal=0.1,
                                                farVal=10.0)
    (_, _, rgba_img, depth_img, seg_img) = p.getCameraImage(width=960,
                                        height=720,
                                        viewMatrix=view_matrix,
                                        projectionMatrix=proj_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.array(rgba_img, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (720,960, 4))        
    rgb_array = rgb_array[:, :, :3]

    return rgb_array, depth_img, seg_img
    

def main(args):

    control_dt = 1./240.
    state_t = 0.
    current_state = 0
    cycle = 0
    success_counter = 0 
    fail_counter = 0
    num_simulations = args.num_sim
    state_durations = [0.2, 0.2, 0.2, 0.2] # number of seconds to hold state

    p.connect(p.GUI)
    p.setGravity(0,0,-10)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
    p.setTimestep = control_dt


    urdfRootPath=pybullet_data.getDataPath()
    pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
    tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
    trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

    
    models = md.model_lib()
    namelist = models.model_name_list
    state_object= [0.7,0,0.1]
    object_name = args.object
    folder_path = "grasp_" + object_name
    objectUid = p.loadURDF(models[object_name], basePosition=state_object)     
    coeff = random.uniform( -math.pi/6., 2*math.pi )


    while cycle < num_simulations:
        
        state_t += control_dt
        
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
        
        if current_state == 0 and state_t > 0.2: # add extra time delay to wait for object to be situated
            move_arm_and_rotate(pandaUid, coeff)
            
        if current_state == 1:        
            approach_object(pandaUid)
            rgb_array, depth_img, seg_img = get_grasp_img()
        
        if current_state == 2:        
            grip_object(pandaUid)
        
        if current_state == 3:
            return_object_to_original_robot_pose(pandaUid)
            
        

        if state_t > state_durations[current_state]:
            
            if current_state == 3: 
                contact = p.getContactPoints(objectUid, pandaUid)    
                if len(contact) > 0: 
                    file_name_success = "success_" + object_name + "_" +  str(success_counter) + ".jpg"
                    file_name_success_depth = "success_depth_" + object_name + "_" +  str(success_counter) + ".jpg"
                    file_name_success_seg = "success_seg_" + object_name + "_" +  str(success_counter) + ".jpg"
                    
                    
                    full_path = os.path.join(folder_path, file_name_success)
                    full_path_seg = os.path.join(folder_path, file_name_success_seg)
                    full_path_depth = os.path.join(folder_path, file_name_success_depth)
                    

                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(full_path, rgb_array)
                    cv2.imwrite(full_path_seg, seg_img)
                    cv2.imwrite(full_path_depth, depth_img)
                    
                    success_counter += 1
                else: 
                    file_name_fail = "fail_" + object_name + "_" +  str(fail_counter) + ".jpg"
                    file_name_fail_depth = "success_depth_" + object_name + "_" +  str(success_counter) + ".jpg"
                    file_name_fail_seg = "success_seg_" + object_name + "_" +  str(success_counter) + ".jpg"
                    
                    full_path = os.path.join(folder_path, file_name_fail)
                    full_path_seg = os.path.join(folder_path, file_name_fail_seg)
                    full_path_depth = os.path.join(folder_path, file_name_fail_depth)


                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(full_path, rgb_array)
                    cv2.imwrite(full_path_seg, seg_img)
                    cv2.imwrite(full_path_depth, depth_img)
                    fail_counter += 1
            
            
            current_state += 1
            
            if current_state >= len(state_durations):
                current_state = 0
                cycle += 1
                reset_object(objectUid, models, object_name, state_object)
                coeff = random.uniform( -math.pi/6., 2*math.pi )
            
            state_t = 0
        


        p.stepSimulation()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='BinaryCodeNet With Local Features')
    parser.add_argument('--num_sim', type=int) 
    parser.add_argument('--object', type=str) 
    args = parser.parse_args()
    main(args)
