import gym
import numpy as np
import time
import os
import sys
# hide diagnostic output
with open(os.devnull, 'w') as devnull:
    # suppress stdout
    orig_stdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    import pybullet
    os.fsync(devnull.fileno())
    os.dup2(orig_stdout_fno, 1)  # restore stdout'''


import pybullet
import pybullet_data
import numpy as np
from PIL import Image
from IPython.display import display


# inspired by https://github.com/dmitrySorokin/ur10_robot/blob/master/ur10_env.py
# and by panda-gym

class UR10(gym.Env):
    

    def __init__(self, is_train=True, distance_threshold=0.15, is_dense=True, angle_control=False, is_fixed=False, 
                 force=0.05, complex_obs_space=False, complex_reward=False, pos_range=0.5,
                 max_steps=500, space='cube'):
        if complex_obs_space:
            self.observation_space = gym.spaces.Dict(dict(
                target=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype='float32'),
                distance=gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float32'),
                joint_angles=gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype='float32'),
                joint_positions=gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype='float32'),
            ))
        else:
            self.observation_space = gym.spaces.Dict(dict(
                target=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype='float32'),
                distance=gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float32'),
            ))

        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        if angle_control:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(5,), dtype='float32')
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype='float32')
        
        self.connect(is_train)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.is_dense = is_dense
        self.distance_threshold = distance_threshold

        self.planeId = None
        self.robot = None
        self.joints = []
        self.links = {}
        self.angle_control = angle_control
        self.ee_link = None
        self.is_fixed = is_fixed
        self.force = force
        self.complex_obs_space = complex_obs_space
        self.complex_reward = complex_reward
        self.prev_dist = None
        self.orig_dist = None
        self.pos_range = pos_range
        self.max_steps = max_steps
        self.space = space

        self.step_id = None
        self.object = None
       
    def _set_goal(self, target_position):
        self.target_position = target_position
        
        d = np.linalg.norm(self.target_position - np.array([0, 0, 1]))
        
        print(d)
        
    def step(self, action):
        self.step_id += 1
        
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high )
        
        if self.angle_control == False:
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:6]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
        
        # self.gripper_value = 1 if action[3] > 0 else -1

        # end effector points down, not up (in case useOrientation==1)
        # self.move_hand(target_pos, self.gripper_orientation, self.gripper_value)
        # print(target_arm_angles)
        self.control_joints(target_arm_angles)

        pybullet.stepSimulation(physicsClientId=self.server)

        # object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        distance = self._get_distance()
        
        return self._get_obs(), self._get_reward(distance), self.is_done(distance), self._get_info(distance, action)
    
    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        ee_displacement = ee_displacement[:3] * self.force  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        
        target_arm_angles = pybullet.calculateInverseKinematics(
            self.robot, self.ee_link, target_ee_position, np.array([1.0, 0.0, 0.0, 0.0]), physicsClientId=self.server
            # maxNumIterations=100, residualThreshold=.01
        )
        
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 6 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * self.force  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_state(i) for i in range(6)])
        
        # print(current_arm_joint_angles, '+', arm_joint_ctrl)
        target_arm_angles = current_arm_joint_angles
        target_arm_angles[:5] += arm_joint_ctrl
        
        return target_arm_angles
    
   
    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_state(self.ee_link)[0]
    
    def get_link_state(self, linkID):
        link_position, _, _, _, _, _, link_velocity, _ = \
            pybullet.getLinkState(self.robot, linkIndex=linkID, computeLinkVelocity=True, physicsClientId=self.server)
        
        return link_position, link_velocity
    
    def get_joint_state(self, jointID):
        joint_position, _, _, _ = pybullet.getJointState(
            self.robot, jointIndex=self.joints[jointID]['jointID'], physicsClientId=self.server
        )
        
        return joint_position

    def control_joints(self, target_angles): #,gripper_value):
        for joint_id in range(6):
            pybullet.setJointMotorControl2(
                self.robot, self.joints[joint_id]['jointID'],
                pybullet.POSITION_CONTROL,
                # pybullet.TORQUE_CONTROL,
                targetPosition=target_angles[joint_id],
                physicsClientId=self.server
            )

    def _reset_world(self, verbose=False):
        pybullet.resetSimulation(physicsClientId=self.server)

        pybullet.setGravity(0, 0, -10, physicsClientId=self.server)
        self.planeId = pybullet.loadURDF('plane.urdf', physicsClientId=self.server)
        # robot_position = [0, 0, 1.0] # fixed start position
        robot_position = [0, 0, 1.0]
        robot_orientation = pybullet.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.server)
        self.robot = pybullet.loadURDF('./ur10mine.urdf', robot_position, robot_orientation,physicsClientId=self.server)

        joint_type = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        self.joints = []
        self.links = {}

        for joint_id in range(pybullet.getNumJoints(self.robot, physicsClientId=self.server)):
            info = pybullet.getJointInfo(self.robot, joint_id, physicsClientId=self.server)
            data = {
                'jointID': info[0],
                'jointName': info[1].decode('utf-8'),
                'jointType': joint_type[info[2]],
                'jointLowerLimit': info[8],
                'jointUpperLimit': info[9],
                'jointMaxForce': info[10],
                'jointMaxVelocity': info[11]
            }
            
            if verbose:
                print('>', joint_id, data)
                print(data['jointType'], joint_id, data['jointName'])
            
            if data['jointType'] != 'FIXED':
                self.joints.append(data)
            
            self.links[data['jointName']] = joint_id
                
            # FIXED: base_link-base_link_inertia
            # FIXED: wrist_3_link-ft_frame
            # FIXED: flange-tool0
            # FIXED: base_link-base_fixed_joint
            # links connected by joints

        self.step_id = 0
        self.object = None
        
        if verbose:
            print('links:', self.links)
        # self.ee_link = self.links['wrist_3-flange']
        self.ee_link = self.links['flange-tool0']

        pybullet.loadURDF('table/table.urdf', basePosition=[0.5, 0, 0], globalScaling=1, physicsClientId=self.server)

        
    def _get_obs(self):
        # Убрал все связанное с ориентацией
        # state = np.zeros(3 * 4 + 3 * 4 + 1)
        state = np.zeros(3 * 2, dtype=np.float32)

        flange_position, _, _, _, _, _, flange_velocity, _ = \
            pybullet.getLinkState(self.robot, linkIndex=self.ee_link, computeLinkVelocity=True, physicsClientId=self.server)
        
        state[:3] = flange_position
        state[3:6] = flange_velocity
        
        if self.complex_obs_space:
            joint_angles = np.zeros(len(self.joints) - 1, dtype=np.float32)
            joint_positions = np.zeros(3 * (len(self.joints) - 1), dtype=np.float32)

            for i in range(len(self.joints) - 1):
                joint_angles[i] = self.get_joint_state(i)
                joint_positions[i*3:(i+1)*3] = self.get_link_state(self.joints[i]['jointID'])[0]

            return {'observation': state, 'target': self.target_position.astype('float32'), 'distance':self._get_distance().astype('float32'), 
                   'joint_angles':joint_angles, 'joint_positions':joint_positions}
        
        else:
            return {'observation': state, 'target': self.target_position.astype('float32'), 'distance':self._get_distance().astype('float32')}
    
    def start_log_video(self, filename):
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, filename, physicsClientId=self.server)

    def stop_log_video(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, physicsClientId=self.server)

    def render(self, mode='human'):
        pass
    
    def __del__(self):
        pybullet.disconnect(physicsClientId=self.server)
        
    def _get_distance(self):
        flange_position, _, _, _, _, _, flange_velocity, _ = \
                pybullet.getLinkState(self.robot, linkIndex=self.ee_link, computeLinkVelocity=True, physicsClientId=self.server)
        
        distance = np.linalg.norm(flange_position - self.target_position)
        
        return distance.astype('float32')
    
    def reset(self, verbose=False, target_pos=None):
        self._reset_world(verbose)
        self.prev_dist = None
        self.orig_dist = None
        
        # x_lims = (0.25, 0.75)
        # y_lims = (-0.35, 0.35)
        # z_lims = (0.8, 1.9)
        # x_lims = (-1.0, 1.0)
        # y_lims = (-1.0, 1.0)
        # z_lims = (1.0, 1.9)
        
        # x_lims = (-self.pos_range/2, self.pos_range/2)
        # y_lims = (-self.pos_range/2, self.pos_range/2)
        # z_lims = (1.0, 1.0 + self.pos_range)
        
        
        if self.space == 'cube':
            x_lims = (0.4, 0.8)
            y_lims = (-0.4, 0.4)
            z_lims = (0.7, 1.3)
            
            x_position = np.random.uniform(*x_lims)
            y_position = np.random.uniform(*y_lims)
            z_position = np.random.uniform(*z_lims)
            
        elif self.space == 'sphere':
            x_lims = (-self.pos_range/2, self.pos_range/2)
            y_lims = (-self.pos_range/2, self.pos_range/2)
            z_lims = (1.0, 1.0 + self.pos_range)
            
            while True:
                x_position = np.random.uniform(*x_lims)
                y_position = np.random.uniform(*y_lims)
                z_position = np.random.uniform(*z_lims)

                d = np.sqrt(x_position ** 2 + y_position ** 2 + (z_position - 1)**2)

                if d < 1.0 and d>0.1:
                    # print('dist:', d)
                    break
        
        else:
            print('ERROR')
        
        
        
       
            
            
        # self.target_position = np.array([x_position, y_position, 0.6])
        self.target_position = np.array([x_position, y_position, z_position])
        
        if target_pos is not None:
            self._set_goal(target_pos)
        
        if self.is_fixed:
            self.target_position = np.array([ 0.3339450, -0.70683447,  1.82771452])
        
        if verbose:
            print('TARGET_POSITION:', self.target_position)

        orientation = pybullet.getQuaternionFromEuler([0, 0, np.pi / 2], physicsClientId=self.server)

        
        pybullet.stepSimulation(physicsClientId=self.server)

        return self._get_obs()

    

    def _get_reward(self, distance):
        if self.is_dense:
            if self.complex_reward:
                # r1 = np.exp(-0.99 * distance) - 1
                r1 = -distance
                
                if self.orig_dist is None:
                    self.orig_dist = distance
                    
                    
                if distance < self.distance_threshold:
                    r2 = 5
                elif distance > self.orig_dist:
                    r2 = -2
                else:
                    r2 = 0
                    
                self.prev_dist = distance
                
                return 4 * r1 + r2
            else:
                return -distance
        else:
            return -(distance > self.distance_threshold).astype(np.float32)

    def is_done(self, distance):
        return self.step_id == self.max_steps or distance <= self.distance_threshold

    def _get_info(self, distance, last_action):
        return {
            'is_success': distance < self.distance_threshold,
            'last_action': last_action
        }

    def connect(self, is_train):        
        if is_train:
            self.server = pybullet.connect(pybullet.DIRECT)
        else:
            self.server = pybullet.connect(pybullet.GUI)
        
    def __del__(self):
        pybullet.disconnect(self.server)


def visualize(ur):
    width = 720
    height = 600
    img_arr = pybullet.getCameraImage(
        width,
        height,
        viewMatrix=pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 1, 1],
            distance=4,
            yaw=0,
            pitch=-10,
            roll=0,
            upAxisIndex=2,
        ),
        projectionMatrix=pybullet.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.01,
            farVal=100,
        ),
        shadow=True,
        lightDirection=[1, 1, 1],
        physicsClientId=ur.server
    )

    width, height, rgba, depth, mask = img_arr
    # print(f"rgba shape={rgba.shape}, dtype={rgba.dtype}")
    display(Image.fromarray(rgba, 'RGBA'))