import argparse
import math
import os
import random
from time import sleep, time
from urllib import robotparser

import cv2
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import urdf_models.models_data as md

# move_arm_and_rotate : this function would set the robotic arm at a random angle before initiating to approach the object
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

# approach_object: this function initiates the robotic arm to move towards the object 
def approach_object(pandaUid):
    p.setJointMotorControl2(pandaUid, 1, 
                            p.POSITION_CONTROL,math.pi/4.+.15)
    p.setJointMotorControl2(pandaUid, 3, 
                            p.POSITION_CONTROL,-math.pi/2.+.15)

# grip_object: this function executes the robotic arm fingers to inch towards grasping the object
def grip_object(pandaUid):
    p.setJointMotorControl2(pandaUid, 9, 
                            p.POSITION_CONTROL, 0.0, force = 200)
    p.setJointMotorControl2(pandaUid, 10, 
                            p.POSITION_CONTROL, 0.0, force = 200)

    # trying to record coordinates of the robot during gripping                    
    
    # initializing a dictionary to compute the robot coordinates during pre-grasp/post-grasp
    joint_dict = {}
    
    # fetch the relevant joint info for each of the joints in the robotic arm
    for joint_num in range(11):
        induvidual_joint_data={}
        joint_info = p.getJointState(pandaUid,joint_num)
        induvidual_joint_data['joint_position'] = joint_info[0]
        induvidual_joint_data['joint_velocity'] = joint_info[1]
        induvidual_joint_data['joint_reaction_forces'] = joint_info[2]
        induvidual_joint_data['applied_joint_motor_torque'] = joint_info[3]
        joint_dict['joint_'+str(joint_num)] = induvidual_joint_data
    
    joint_dict['link_state'] = p.getLinkState(pandaUid, 11)[0]

    return joint_dict

# return_object_to_original_robot_pose: this function allows the robotic arm to come back to original position after attempting to grasp the underlying object.
def return_object_to_original_robot_pose(pandaUid): 
    p.setJointMotorControl2(pandaUid, 1, 
                            p.POSITION_CONTROL,math.pi/4.-1)
    p.setJointMotorControl2(pandaUid, 3, 
                            p.POSITION_CONTROL,-math.pi/2.-1)

# reset_object: this function resets the object to grasp for successive simulations
def reset_object(objectUid, models, object_name, state_object): 
    p.removeBody(objectUid)
    objectUid =  p.loadURDF(models[object_name], basePosition=state_object)
    return objectUid

# get_grasp_img: this function facilitates in fetching the snapshots at various instances during the grasp.
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

    #depth_img = (depth_img*255).astype(np.uint8)
    depth_new = np.array(depth_img,dtype=np.uint8)

    return rgb_array, depth_new, seg_img
    

def main(args):

    # setting the initial parameters.
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

    # setting empty arrays to hold coordinates and contact information
    pre_grasp_robot_coordinates = []
    post_grasp_robot_coordinates = []
    contact_points_9 = []
    contact_points_10 = []
    
    # loading urdf of robotic arm and the object under consideration.
    urdfRootPath=pybullet_data.getDataPath()
    print("Root path: \n",urdfRootPath)
    pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
    tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
    trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

    
    models = md.model_lib()
    namelist = models.model_name_list
    state_object= [0.7,0,0.1]
    object_name = args.object
    folder_path = args.folder_path
    objectUid = p.loadURDF(models[object_name], basePosition=state_object)     
    coeff = random.uniform( -math.pi/6., 2*math.pi )

    # creating dataframe to store coordinate and result image info
    df = pd.DataFrame(columns=['pre_grasp_robot_coordinates','post_grasp_robot_coordinates','contact_points_left_finger','contact_points_right_finger','length_of_contact_points','image_snapshot_path'])

    # for every simulations
    while cycle < num_simulations:
        
        state_t += control_dt
        
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
        
        if current_state == 0 and state_t > 0.2: # add extra time delay to wait for object to be situated
            move_arm_and_rotate(pandaUid, coeff)
            
        # approach the object and the final snapshot before the object begins to initate gripping is updated    
        if current_state == 1:        
            approach_object(pandaUid)
            rgb_array, depth_img, seg_img = get_grasp_img()
        
        # fetch contact point cloud of the fingers of the robot (joint 9 and joint 10)
        if current_state == 2:
            grip_contact = p.getContactPoints(objectUid, pandaUid)
            
            # fetching pre-grasp and post-grasp coordinates of all joints of robot
            get_robot_coordinates = grip_object(pandaUid)

            # if there was no capture of robot coordinates earlier, then it is the first (pre-grasp instance)
            if len(pre_grasp_robot_coordinates) == 0:
                pre_grasp_robot_coordinates = get_robot_coordinates
            else:
                post_grasp_robot_coordinates = get_robot_coordinates
            
            # we ought to fetch contact_points_left and contact_points_right separately 
            for contact_information in grip_contact:
                # print("\n New contact_information: ",contact_information)
                if round(abs(contact_information[8]),3) <= 0.001: # if distance is around 1e-3
                    contact_info ={}
                    if contact_information[4] == 10: #if contact info is of right finger, add to right_fingers' point cloud
                        contact_info['position_on_A'] = contact_information[5]
                        contact_info['position_on_B'] = contact_information[6]
                        contact_info['contact_normal_on_B'] = contact_information[7]
                        contact_info['contact_distance'] = contact_information[8]
                        contact_info['Normal_force'] = contact_information[9]
                        contact_points_10.append(contact_info)
                    if contact_information[4] == 9: # add to left_fingers' point_cloud
                        contact_info['position_on_A'] = contact_information[5]
                        contact_info['position_on_B'] = contact_information[6]
                        contact_info['contact_normal_on_B'] = contact_information[7]
                        contact_info['contact_distance'] = contact_information[8]
                        contact_info['Normal_force'] = contact_information[9]
                        contact_points_9.append(contact_info)
            
            # print("\n For current simulation: \n")
            # print("\n Left finger point_cloud: ",contact_points_9)
            # print("\n Right ginger point_cloud: ",contact_points_10)
        
        # return back to original pose 
        if current_state == 3:
            return_object_to_original_robot_pose(pandaUid)

        if state_t > state_durations[current_state]:
            
            if current_state == 3: # fetch contact points and check if the grasp is successful
                return_contact = p.getContactPoints(objectUid, pandaUid)   
                
                if len(return_contact) >= 5: # check success of the grasp
                    file_name_success = "success_" + object_name + "_" +  str(success_counter) + ".jpg"
                    file_name_success_depth = "success_depth_" + object_name + "_" +  str(success_counter) + ".jpg"
                    file_name_success_seg = "success_seg_" + object_name + "_" +  str(success_counter) + ".jpg"
                    
                    
                    full_path = os.path.join(folder_path, file_name_success)
                    full_path_seg = os.path.join(folder_path, file_name_success_seg)
                    full_path_depth = os.path.join(folder_path, file_name_success_depth)
                    

                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)

                    # store the snapshot output
                    cv2.imwrite(full_path, rgb_array)
                    cv2.imwrite(full_path_seg, seg_img)
                    cv2.imwrite(full_path_depth, depth_img)
                    success_counter += 1

                else: 
                    file_name_fail = "fail_" + object_name + "_" +  str(fail_counter) + ".jpg"
                    file_name_fail_depth = "fail_depth_" + object_name + "_" +  str(success_counter) + ".jpg"
                    file_name_fail_seg = "fail_seg_" + object_name + "_" +  str(success_counter) + ".jpg"
                    
                    full_path = os.path.join(folder_path, file_name_fail)
                    
                    full_path_seg = os.path.join(folder_path, file_name_fail_seg)
                    
                    full_path_depth = os.path.join(folder_path, file_name_fail_depth)
                    
                    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(full_path, rgb_array)
                    print("saving seg file")
                    cv2.imwrite(full_path_seg, seg_img)
                    print("saving depth file")
                    cv2.imwrite(full_path_depth, depth_img)
                    fail_counter += 1
                
                # store length information of contact left/right robot fingers 
                length_of_contact_points={}
                length_of_contact_points['contact_joint_left_finger'] = len(contact_points_9)
                length_of_contact_points['contact_joint_right_finger'] = len(contact_points_10)

                # adding new datapoint to data frame
                df.loc[len(df.index)] = [pre_grasp_robot_coordinates,post_grasp_robot_coordinates,contact_points_9,contact_points_10,length_of_contact_points,full_path]
                
            current_state += 1
            
            if current_state >= len(state_durations):
                current_state = 0
                cycle += 1
                reset_object(objectUid, models, object_name, state_object)
                coeff = random.uniform( -math.pi/6., 2*math.pi )

                # resetting arrays for capturing next simulation
                pre_grasp_robot_coordinates = []
                post_grasp_robot_coordinates = []
                contact_points_9 = []
                contact_points_10 = []
            
            state_t = 0
        


        p.stepSimulation()

        # write the dataframe to excel
        output_file_path="output_data"
        os.makedirs(output_file_path, exist_ok=True)
        output_json_path = output_file_path+"/"+object_name+"_data.json"
        df.to_json(output_json_path, orient='index')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='BinaryCodeNet With Local Features')
    parser.add_argument('--num_sim', type=int) 
    parser.add_argument('--object', type=str) 
    parser.add_argument('--folder_path', type=str)
    args = parser.parse_args()
    main(args)