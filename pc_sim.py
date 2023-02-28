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

# get_grasp_img: this function facilitates in fetching the snapshots at various instances during the grasp.
def get_grasp_img():
    
    #Camera Refenrence Frame Because We Give Camera Position Then Photo is Captured There   
    
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
    
    # Generate Point-Cloud Here



    rgb_array = np.array(rgba_img, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (720,960, 4))        
    rgb_array = rgb_array[:, :, :3]

    #depth_img = (depth_img*255).astype(np.uint8)
    depth_new = np.array(depth_img,dtype=np.uint8)

    return rgb_array, depth_new, seg_img


def get_point_cloud(width, height, view_matrix, proj_matrix):
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    # get a depth image
    # "infinite" depths will have a value close to 1
    image_arr = pb.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    depth = image_arr[3]

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points



def get_point_cloud_image(): 

    #Camera Refenrence Frame Because We Give Camera Position Then Photo is Captured There
    
    
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.55,0.65,-0.64],
                                                                distance=1.80,
                                                                yaw=8.40,
                                                                pitch=-47.6,
                                                                roll=0,
                                                                upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                aspect=float(960) /720,
                                                nearVal=0.1,
                                                farVal=10.0)
    
    height=720 
    width = 960
    
    
    (_, _, rgba_img, depth_img, seg_img) = p.getCameraImage(width=960,
                                        height=720,
                                        viewMatrix=view_matrix,
                                        projectionMatrix=proj_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth_img.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    
    
    rgb_array = np.array(rgba_img, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (720,960, 4))        
    rgb_array = rgb_array[:, :, :3]

    #depth_img = (depth_img*255).astype(np.uint8)
    depth_new = np.array(depth_img,dtype=np.uint8)

    return rgb_array, depth_new, seg_img, points





def main(args):

    # setting the initial parameters.
    control_dt = 1./240.
    state_t = 0.
    current_state = 0
    cycle = 0    
    num_simulations = args.num_sim
    
    p.connect(p.GUI)
    p.setGravity(0,0,-10)
    p.resetDebugVisualizerCamera(cameraTargetPosition=[0.55,0.65,-0.64],
                                                                cameraDistance=1.60,
                                                                cameraYaw=8.4,
                                                                cameraPitch=-47.6,
                                                                )
    p.setTimestep = control_dt

    # loading urdf of robotic arm and the object under consideration.
    urdfRootPath=pybullet_data.getDataPath()

    print("Root path: \n",urdfRootPath)
    #pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
    #tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
    #trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
    p.loadURDF(os.path.join(urdfRootPath, 'plane_transparent.urdf')) # load the floor of the object
        
    models = md.model_lib()
    namelist = models.model_name_list
    print(namelist)
    state_object= [0.7,0,0.1]
    object_name = args.object
    folder_path = args.folder_path
    
   
    objectUid = p.loadURDF(models[object_name], basePosition=state_object)      #p.loadURDF(os.path.join('cheezit', 'model.urdf'), basePosition=state_object)  #
    
    
    while cycle < num_simulations:       
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
        if cycle == num_simulations-1: # add extra time delay to wait for object to be situated            
            rgb_array, depth_img, seg_img, point_cloud = get_point_cloud_image()  
            output_file_path="output_data"
            os.makedirs(output_file_path, exist_ok=True)
            output_csv_path = output_file_path+"/"+object_name+"_point_cloud.csv"
            pd.DataFrame(point_cloud).to_csv(output_csv_path, header=False, index=False)
        p.stepSimulation()
        cycle += 1
           

        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='BinaryCodeNet With Local Features')
    parser.add_argument('--num_sim', type=int) 
    parser.add_argument('--object', type=str) 
    parser.add_argument('--folder_path', type=str)
    args = parser.parse_args()
    main(args)