from math import *
import math
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate, KinovaRobotTemplate
)


class Kinova(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()

    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            joint_values (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        curr_joint_values = joint_values.copy()

        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
        
        # DH parameters for each joint
        DH = np.zeros((self.num_dof + 1, 4))
        DH[0] = [0, 0, 0, pi]
        DH[1] = [curr_joint_values[0], -self.l1 - self.l2, 0, pi/2]
        DH[2] = [curr_joint_values[1] - pi/2, 0, self.l3, pi]
        DH[3] = [curr_joint_values[2] - pi/2, 0, 0, pi/2]
        DH[4] = [curr_joint_values[3], -self.l4 - self.l5, 0, -pi/2]
        DH[5] = [curr_joint_values[4], 0, 0, pi/2]
        DH[6] = [curr_joint_values[5], -self.l6 - self.l7, 0, pi]
        

        # Compute the transformation matrices
        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist
    
    def calc_inverse_kinematics(self, ee, init_joint_values, soln=0):
        #currently does not account for multiple solns
        #actually change the calcs for different thetas
        d5 = self.l4 + self.l5
        H0_6 = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))
        w_x = ee.x - d5*H0_6[0,3]
        w_y = ee.y - d5*H0_6[1,3]
        w_z = ee.z - d5*H0_6[2,3]
        
        theta1a = math.atan2(ee.x, ee.y)
        theta1b = math.atan2(ee.x, ee.y)
        
        r = math.sqrt(w_x**2 + w_y**2)
        s = w_z - self.l1
        theta3a = math.arccos((r**2 + s**2 - self.l3**2 - self.l4**2)/(2*self.l3*self.l4))
        theta3b = math.arccos((r**2 + s**2 - self.l3**2 - self.l4**2)/(2*self.l3*self.l4))
        theta2a = math.arcsin(((self.l3 + self.l4*math.cos(theta3))*s - self.l4*math.sin(theta3)*r)/(r**2 + s**2))
        theta2b= math.arccos(((self.l3 + self.l4*math.cos(theta3))*r - self.l4*math.sin(theta3)*s)/(r**2 + s**2))
        
        if soln == 0:
            theta1 = theta1a
            theta2 = theta2a
            theta3 = theta3a

        elif soln == 1:
            theta1 = theta1a
            theta2 = theta2b
            theta3 = theta3b

        elif soln == 2:
            theta1 = theta1b
            theta2 = theta2a
            theta3 = theta3a

        elif soln == 3:
            theta1 = theta1b
            theta2 = theta2b
            theta3 = theta3b
        
        R0_1 = self.Rz(theta1)@self.Ry(theta1)@self.Rx(theta1)
        R1_2 = self.Rz(theta2)@self.Ry(theta2)@self.Rx(theta2)
        R2_3 = self.Rz(theta3)@self.Ry(theta3)@self.Rx(theta3)
        R0_3 = R0_1@R1_2@R2_3
        R3_6 = R0_3.T @ R0_3
    
        theta5 = math.atan2(math.atan2(1-(math.sin(theta1)*H0_6[0])))
        
    
        
        
    


if __name__ == "__main__":
    model = Kinova()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()