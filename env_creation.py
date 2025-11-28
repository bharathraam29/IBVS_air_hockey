import pybullet as p
import pybullet_data
import numpy as np
import time
import os


""" CAMERA SETTINGS """

camera_width = 512                                             #image width
camera_height = 512                                            #image height
camera_fov = 120                                                #field of view of camera
camera_focal_depth = 0.5*camera_height/np.tan(0.5*np.pi/180*camera_fov) 
                                                               #focal depth in pixel space
camera_aspect = camera_width/camera_height                     #aspect ratio
camera_near = 0.02                                             #near clipping plane in meters, do not set non-zero
camera_far = 100                                               #far clipping plane in meters

    # Robot with Camera Class
class eye_in_hand_robot:
    def get_ee_position(self):
        '''
        Function to return the end-effector of the link. This is the very tip of the robot at the end of the jaws.
        '''
        endEffectorIndex = self.numActiveJoints
        endEffectorState = p.getLinkState(self.robot_id, endEffectorIndex)
        endEffectorPos = np.array(endEffectorState[0])
        endEffectorOrn = np.array(p.getMatrixFromQuaternion(endEffectorState[1])).reshape(3,3)
        
        #add an offset to get past the forceps
        endEffectorPos += self.camera_offset*endEffectorOrn[:,2]
        return endEffectorPos, endEffectorOrn

    def get_current_joint_angles(self):
        # Get the current joint angles
        joint_angles = np.zeros(self.numActiveJoints)
        for i in range(self.numActiveJoints):
            joint_state = p.getJointState(self.robot_id, self._active_joint_indices[i])
            joint_angles[i] = joint_state[0]
        return joint_angles
    
    def get_jacobian_at_current_position(self):
        #Returns the Robot Jacobian of the last active link
        mpos, mvel, mtorq = self.get_active_joint_states()   
        zero_vec = [0.0]*len(mpos)
        linearJacobian, angularJacobian = p.calculateJacobian(self.robot_id, 
                                                              self.numActiveJoints,
                                                              [0,0,self.camera_offset],
                                                              mpos, 
                                                              zero_vec,
                                                              zero_vec)
        #only return the active joint's jacobians
        Jacobian = np.vstack((linearJacobian,angularJacobian))
        return Jacobian[:,:self.numActiveJoints]
    
    def set_joint_position(self, desireJointPositions, kp=1.0, kv=0.3):
        '''Set  the joint angle positions of the robot'''
        zero_vec = [0.0] * self._numLinkJoints
        allJointPositionObjectives = [0.0]*self._numLinkJoints
        for i in range(desireJointPositions.shape[0]):
            idx = self._active_joint_indices[i]
            allJointPositionObjectives[idx] = desireJointPositions[i]

        p.setJointMotorControlArray(self.robot_id,
                                    range(self._numLinkJoints),
                                    p.POSITION_CONTROL,
                                    targetPositions=allJointPositionObjectives,
                                    targetVelocities=zero_vec,
                                    positionGains=[kp] * self._numLinkJoints,
                                    velocityGains=[kv] * self._numLinkJoints)

    def get_active_joint_states(self):
        '''Get the states (position, velocity, and torques) of the active joints of the robot
        '''
        joint_states = p.getJointStates(self.robot_id, range(self._numLinkJoints))
        joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques


         
    def __init__(self, robot_id, initialJointPos):
        self.robot_id = robot_id
        self.eeFrameId = []
        self.camera_offset = 0.1 #offset camera in z direction to avoid grippers
        # Get the joint info
        self._numLinkJoints = p.getNumJoints(self.robot_id) #includes passive joint
        jointInfo = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        
        # Get joint locations (some joints are passive)
        self._active_joint_indices = []
        for i in range(self._numLinkJoints):
            if jointInfo[i][2]==p.JOINT_REVOLUTE:
                self._active_joint_indices.append(jointInfo[i][0])
        self.numActiveJoints = len(self._active_joint_indices) #exact number of active joints

        #reset joints
        for i in range(self._numLinkJoints):
            p.resetJointState(self.robot_id,i,initialJointPos[i])


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

    #adjust for clipping and nonlinear distance i.e., 1/d (0 is closest, i.e., near, 1 is furthest away, i.e., far
    depthImgLinearized =camera_far*camera_near/(camera_far+camera_near-(camera_far-camera_near)*nonlinDepthImg)

    #convert to numpy and a rgb-d image
    rgb_image = np.array(rgbImg[:,:,:3], dtype=np.uint8)
    depth_image = np.array(depthImgLinearized, dtype=np.float32)
    return rgb_image, depth_image
 

def crossed_line(sphere_position, sphere_velocity):
    if sphere_position[0] < -0.75:
        return True
    return False

# Start the connection to the physics server
physicsClient = p.connect(p.GUI)#(p.DIRECT)
time_step = 0.001
p.resetSimulation()
p.setTimeStep(time_step)
p.setGravity(0, 0, -9.8)

# Set the path to the URDF files included with PyBullet
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load a plane URDF
p.loadURDF('plane.urdf')


# adding robot into the environment
''' Create Robot Instance'''
pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)
p.resetBasePositionAndOrientation(pandaUid, [0, 0, 0], [0, 0, 0, 1])
initialJointPosition = [0,-np.pi/4,np.pi/4,-np.pi/4,np.pi/4,np.pi/4,np.pi/4,0,0,0,0,0]
robot = eye_in_hand_robot(pandaUid,initialJointPosition)
p.stepSimulation() # need to do this to initialize robot

# Place obstacles
box_length = 2
box_width = 0.2
box_depth = 0.5
right_object_center = [0.75, 1, 0.01]
right_object_orientation = [0, 0, np.pi/2]
right_object_color = [0.8, 0.0, 0.0, 1]
geomBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_depth/2])
visualBox = p.createVisualShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_depth/2], rgbaColor=right_object_color)
right_boxId = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=geomBox,
    baseVisualShapeIndex=visualBox,
    basePosition=np.array(right_object_center),
    baseOrientation=p.getQuaternionFromEuler(right_object_orientation)
)   
right_ObjectModelPos, right_modelOrn  = p.getBasePositionAndOrientation(right_boxId)

left_object_center = [-0.75, 1, 0.01]
left_object_orientation = [0, 0, np.pi/2]
left_object_color = [0.8, 0.0, 0.0, 1]
geomBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_depth/2])
visualBox = p.createVisualShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_depth/2], rgbaColor=left_object_color)
left_boxId = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=geomBox,
    baseVisualShapeIndex=visualBox,
    basePosition=np.array(left_object_center),
    baseOrientation=p.getQuaternionFromEuler(left_object_orientation)
)
left_ObjectModelPos, left_modelOrn  = p.getBasePositionAndOrientation(left_boxId)

# adding a sphere what is going spwan in the middle of the two boxes and roll towards the left box
sphere_radius = 0.1 # Corrected: Reduced from 5 to a visible size (0.1m)

sphere_mass = 1

sphere_position = [0, 1, 0.1]

sphere_orientation = [0, 0, 0]

sphere_color = [0.0, 0.8, 0.8, 1]

sphere_geom = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)

sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=sphere_color)

sphere_id = p.createMultiBody(

    baseMass=sphere_mass,

    baseCollisionShapeIndex=sphere_geom,

    baseVisualShapeIndex=sphere_visual,

    basePosition=np.array(sphere_position),

    baseOrientation=p.getQuaternionFromEuler(sphere_orientation)

)

sphere_ObjectModelPos, sphere_modelOrn  = p.getBasePositionAndOrientation(sphere_id)
#reset debug gui camera position so we can see the robot up close


sphere_velocity = [-10, -5, 0] # Negative X velocity to move towards the left box

p.resetBaseVelocity(
    objectUniqueId=sphere_id,
    linearVelocity=sphere_velocity
)
p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0,0,.5])

# Increase the number of steps to simulate for a longer time (e.g., 2 seconds)
SIMULATION_DURATION_SECONDS = 2.0
NUM_STEPS = int(SIMULATION_DURATION_SECONDS / time_step) # 2.0 / 0.001 = 2000 steps

print(f"Running simulation for {SIMULATION_DURATION_SECONDS} seconds ({NUM_STEPS} steps)")
goal_threshold = 0.25 
line_start = [-1.5, goal_threshold, 0.01] # Start X, Goal Y, Z (slightly above floor)
line_end = [1.5, goal_threshold, 0.01]   # End X, Goal Y, Z (slightly above floor)
line_color = [0.8, 0.0, 0.8]

# Add the persistent debug line
goal_line_id = p.addUserDebugLine(
    lineFromXYZ=line_start,
    lineToXYZ=line_end,
    lineColorRGB=line_color,
    lineWidth=2 # Make it thicker to be visible
)

for ITER in range(NUM_STEPS):
    #get sphere position from the physics server
    sphere_position,_ = p.getBasePositionAndOrientation(sphere_id)
    if sphere_position[1] < goal_threshold:
        print("Sphere has crossed the goal line")
        break
    p.stepSimulation()
    # Add a small delay for better visualization in GUI mode
    # NOTE: In DIRECT mode, you would skip this sleep.
    time.sleep(time_step) # Wait for the time step duration to render the frame

time.sleep(10) # Keep the window open for 10 seconds after the simulation ends
p.disconnect()