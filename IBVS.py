#import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import cv2
from scipy.spatial.transform import Rotation as Rot            #can use this to apply angular rotations to coordinate frames



import imageio

#camera image dimensions
camera_width = 512      #image width
camera_height = 512     #image height

#control objectives (if you wish, you can play with these values for fun)
object_location_desired = np.array([camera_width/2,camera_height/2])  #center the object to middle of image
                                                                
K_p_x = 0.1           #Proportional control gain for translation
K_p_Omega = 0.02      #Proportional control gain for rotation       

MAX_ITER = 200


def getImageJacobian(u_px,v_px,depthImg,focal_length, imgWidth, imgHeight):
    ''' Inputs: 
    u_px, v_px is pixel coordinates from image with top-left corner as 0,0
    depthImg is the depth map
    f is the focal length
    
    Outputs: image_jacobian, a 2x6 matrix'''

    
    
    f = focal_length
    z = depthImg[v_px, u_px]
    c_x = imgWidth/2
    c_y = imgHeight/2

    # Row 1
    J11 = -f / z
    J12 = 0
    J13 = (u_px - c_x) / z
    J14 = ((u_px - c_x) * (v_px - c_y)) / f
    J15 = -f - ((u_px - c_x)**2) / f
    J16 = (v_px - c_y)
    
    # Row 2
    J21 = 0
    J22 = -f / z
    J23 = (v_px - c_y) / z
    J24 = f + ((v_px - c_y)**2) / f
    J25 = -((v_px - c_y) * (u_px - c_x)) / f
    J26 = -(u_px - c_x)
    
    
    image_jacobian = np.array([
        [J11, J12, J13, J14, J15, J16],
        [J21, J22, J23, J24, J25, J26]
    ])

    return image_jacobian
    


def findCameraControl(object_loc_des, object_loc, image_jacobian):
    ''' Inputs:
    object_loc_des: desired [x,y] pixel locations for object
    object_loc: current [x,y] pixel locations as found from computer vision
    image_jacobian: the image jacobian 
    Outputs:
    delta_X: the scaled displacement in position of camera (world frame) to reduce the error
    delta_Omega: the scaled angular velocity of camera (world frame omega-x,y,z) to reduce the error
    
    '''
    
    L_inv = np.linalg.pinv(image_jacobian)
    del_pix = object_loc_des - object_loc
    
    del_cam_frame = L_inv@del_pix

    delta_X = del_cam_frame[0:3] * K_p_x

    delta_Omega = del_cam_frame[3:] * K_p_Omega 

    return delta_X, delta_Omega

    

def findJointControl(robot, delta_X, delta_Omega):
    ''' Inputs:
    delta_X: the scaled displacement in position of camera (world frame) to reduce the error
    delta_Omega: the scaled angular velocity of camera (world frame omega-x,y,z) to reduce the error
    Outputs:
    delta_Q: the change in robot joints to cause the camera to move delta_X and delta_Omega
    '''
    
    J = robot.get_jacobian_at_current_position()

    J_inv =pseudo_inverse(J)

    del_state = np.concatenate([delta_X, delta_Omega])
    del_Q = J_inv@del_state
    
    curr_jointPositions = robot.get_current_joint_angles()  
    new_jointPositions = curr_jointPositions + del_Q
    
    return new_jointPositions



def image_visual_servo():

    frames = []
    for ITER in range(MAX_ITER):
        p.stepSimulation()
        ''' Match Camera Pose to Robot End-Effector and Get Image'''
        
        cameraPosition, eeOrientation = robot.get_ee_position()
        cameraOrientation = eeOrientation # @ Rz 

        rgb, depth = get_camera_img_float(cameraPosition, cameraOrientation)
        
        draw_coordinate_frame(cameraPosition, cameraOrientation, 0.1)

        ''' Magical Computer Vision Algorithm that gets locations of objects in the image, as object_loc (Do Not Remove)'''
        viewMat, projMat = get_camera_view_and_projection_opencv(cameraPosition, cameraOrientation)
        object_loc = opengl_plot_world_to_pixelspace(object_center, viewMat, projMat,camera_width, camera_height)

        # Image Jacobian
        imageJacobian = getImageJacobian(object_loc[0], object_loc[1], depth, camera_focal_depth, camera_width, camera_height)

        # Camera control
        delta_X, delta_Omega = findCameraControl(object_location_desired, object_loc, imageJacobian)
        
        
        delta_X = cameraOrientation @ delta_X # cameraOrientation@
        delta_Omega = cameraOrientation @ delta_Omega # cameraOrientation@
        
        #set Next Joint Targets
        new_jointPositions = findJointControl(robot, delta_X, delta_Omega)
        
        
        r = Rot.from_matrix(eeOrientation)
        
        sing = 1.0/3
        q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

        q_sing = -(robot.get_current_joint_angles() - q_home) # Null space control to operate near to home configuration for jerk free movement
        
        joint_null = (np.eye(7) - J_rob_inv@J_rob)@(roll_null*q_roll+sing*q_sing)

        robot.set_joint_position(new_jointPositions + joint_null)
        
        
        # show image
        u, v = object_loc
        cv2.circle(rgb, (int(camera_width/2), int(camera_height/2)), 5, (0, 255, 0), -1)
        cv2.circle(rgb, (u,v), 5, (255, 0, 255), -1)
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb)
        frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    imageio.mimsave('robot_servoing_stationary_null.gif', frames, fps=20)
    print("Gif saved")

    #close the physics server
    cv2.destroyAllWindows()    
    p.disconnect() 




if __name__ == "__main__":
    image_visual_servo()