#import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import cv2
from scipy.spatial.transform import Rotation as Rot            #can use this to apply angular rotations to coordinate frames

import sympy as sp

import imageio

#camera (don't change these settings)
camera_width = 512                                             #image width
camera_height = 512                                            #image height
camera_fov = 120                                                #field of view of camera

camera_aspect = camera_width/camera_height                     #aspect ratio
camera_near = 0.02                                             #near clipping plane in meters, do not set non-zero
camera_far = 100                                               #far clipping plane in meters


#control objectives (if you wish, you can play with these values for fun)
object_location_desired = np.array([camera_width/2,camera_height/2])  #center the object to middle of image
                                                                
K_p_x = 0.1                                                    #Proportional control gain for translation
K_p_Omega = 0.02                                               #Proportional control gain for rotation       




def pseudo_inverse(J):

    m, n = J.shape
    factor = 1e-2
    
    if m >= n:
        # (JTJ)-1 J
        JTJ = J.T @ J + factor**2*np.eye(J.shape[1])
        return np.linalg.inv(JTJ) @ J.T
    else:
        # JT(JJT)-1
        JJT = J @ J.T + factor**2*np.eye(J.shape[0])
        return J.T @ np.linalg.inv(JJT)


def draw_coordinate_frame(position, orientation, length, frameId = []):
    '''
    Draws a coordinate frame x,y,z with scaled lengths on the axes 
    in a position and orientation relative to the world coordinate frame
    pos: 3-element numpy array
    orientation: 3x3 numpy matrix
    length: length of the plotted x,y,z axes
    frameId: a unique ID for the frame. If this supplied, then it will erase the previous location of the frame
    
    returns the frameId
    '''
    if len(frameId)!=0:
        p.removeUserDebugItem(frameId[0])
        p.removeUserDebugItem(frameId[1])
        p.removeUserDebugItem(frameId[2])
    
    lineIdx=p.addUserDebugLine(position, position + np.dot(orientation, [length, 0, 0]), [1, 0, 0])  # x-axis in red
    lineIdy=p.addUserDebugLine(position, position + np.dot(orientation, [0, length, 0]), [0, 1, 0])  # y-axis in green
    lineIdz=p.addUserDebugLine(position, position + np.dot(orientation, [0, 0, length]), [0, 0, 1])  # z-axis in blue

    return lineIdx,lineIdy,lineIdz

def opengl_plot_world_to_pixelspace(pt_in_3D_to_project, viewMat, projMat, imgWidth, imgHeight):
    ''' Plots a x,y,z location in the world in an openCV image
    This is used for debugging, e.g. given a known location in the world, verify it appears in the camera
    when using p.getCameraImage(...). The output [u,v], when plot with opencv, should line up with object 
    in the image from p.getCameraImage(...)
    '''
    pt_in_3D_to_project = np.append(pt_in_3D_to_project,1)
    #print('Point in 3D to project:', pt_in_3D_to_project)

    pt_in_3D_in_camera_frame = viewMat @ pt_in_3D_to_project
    #print('Point in camera space: ', pt_in_3D_in_camera_frame)

    # Convert coordinates to get normalized device coordinates (before rescale)
    uvzw = projMat @ pt_in_3D_in_camera_frame
    #print('after projection: ', uvzw)

    # scale to get the normalized device coordinates
    uvzw_NDC = uvzw/uvzw[3]
    #print('after normalization: ', uvzw_NDC)

    #x,y specifies lower left corner of viewport rectangle, in pixels. initial value is (0,)
    u = ((uvzw_NDC[0] + 1) / 2.0) * imgWidth
    v = ((1-uvzw_NDC[1]) / 2.0) * imgHeight

    return [int(u),int(v)]

    
def get_camera_view_and_projection_opencv(cameraPos, camereaOrn):
    '''Gets the view and projection matrix for a camera at position (3) and orientation (3x3)'''
    __camera_view_matrix_opengl = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                                   cameraTargetPosition=cameraPos+camereaOrn[:,2],
                                                   cameraUpVector=-camereaOrn[:,1])

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(camera_fov, camera_aspect, camera_near, camera_far)        
    _, _, rgbImg, depthImg, _ = p.getCameraImage(camera_width, 
                                                 camera_height, 
                                                 __camera_view_matrix_opengl,
                                                 __camera_projection_matrix_opengl, 
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    #returns camera view and projection matrices in a form that fits openCV
    viewMat = np.array(__camera_view_matrix_opengl).reshape(4,4).T
    projMat = np.array(__camera_projection_matrix_opengl).reshape(4,4).T
    return viewMat, projMat

def get_camera_img_float(cameraPos, camereaOrn):
    ''' Gets the image and depth map from a camera at a position cameraPos (3) and cameraOrn (3x3) in space. '''
    __camera_view_matrix_opengl = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                                   cameraTargetPosition=cameraPos+camereaOrn[:,2],
                                                   cameraUpVector=-camereaOrn[:,1])

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(camera_fov, camera_aspect, camera_near, camera_far)        
    width, height, rgbImg, nonlinDepthImg, _ = p.getCameraImage(camera_width, 
                                                 camera_height, 
                                                 __camera_view_matrix_opengl,
                                                 __camera_projection_matrix_opengl, 
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Convert to numpy arrays
    rgbImg = np.array(rgbImg, dtype=np.uint8).reshape(height, width, 4)
    nonlinDepthImg = np.array(nonlinDepthImg, dtype=np.float32).reshape(height, width)

    
    #adjust for clipping and nonlinear distance i.e., 1/d (0 is closest, i.e., near, 1 is furthest away, i.e., far
    depthImgLinearized =camera_far*camera_near/(camera_far+camera_near-(camera_far-camera_near)*nonlinDepthImg)

    #convert to numpy and a rgb-d image
    rgb_image = np.array(rgbImg[:,:,:3], dtype=np.uint8)
    depth_image = np.array(depthImgLinearized, dtype=np.float32)
    return rgb_image, depth_image