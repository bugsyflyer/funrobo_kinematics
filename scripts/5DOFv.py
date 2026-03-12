from math import *
import numpy as np
#from sympy import beta
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()

    def compute_transformation_matrices(self, joint_values):
    
        theta = joint_values
        DH = np.array([
            [theta[0], self.l1, 0, -pi/2],
            [theta[1]-(pi/2), 0, self.l2, pi],
            [theta[2], 0, self.l3, pi],
            [theta[3]+(pi/2), 0, 0, pi/2],
            [theta[4], self.l4+self.l5, 0, 0]
        ])

        # compute transformation matrices for each joint
        Hlist = [ut.dh_to_matrix(dh) for dh in DH] 

        # compute cumulative transformations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)
            
        return H_cumulative, Hlist    


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
        for i in range(4):
            theta = curr_joint_values[i]
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])

       
        H_cumulative, Hlist = self.compute_transformation_matrices(curr_joint_values)

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
       
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_inverse_kinematics(self, ee, joint_values: list, radians=True, soln = 0):
        
        # position and rotation of end effector
        p_ee = np.array([ee.x, ee.y, ee.z])
        r_ee = ut.euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
        #wrist center position
        piv_wrist = p_ee - (self.l4 + self.l5) * (r_ee @ np.array([0, 0, 1]))
        wrist_x, wrist_y, wrist_z = piv_wrist[0], piv_wrist[1], piv_wrist[2]
        # calculate theta1 
        theta1 = atan2(wrist_y, wrist_x)
        # create triangle from joint 2, joint 3, and wrist center
        h_dist = sqrt(wrist_x**2 + wrist_y**2) 
        v_dist = wrist_z - self.l1
        hv_hype = h_dist**2 + v_dist**2
        # calculate beta using law of cosines
        l_2 = self.l2
        l_3 = self.l3
        cos_beta = np.clip((l_2**2 + l_3**2 - hv_hype) / (2 * l_2 * l_3), -1, 1)
        beta = acos(cos_beta)
        if soln == 0:
            theta3 = np.pi - beta
        else:
            theta3 = -(np.pi - beta)

        # angular offset 
        alpha = atan2(l_3 * sin(theta3), l_2 + l_3 * cos(theta3))
        # angular trajectory
        gamma = atan2(v_dist, h_dist)
        theta2 = (pi/2) - (gamma - alpha)
        # calculate roatation from 3 to ee
        q_list = [theta1, theta2, theta3, 0, 0]
        H_cumulative, _ = self.compute_transformation_matrices(q_list)
        R3 = H_cumulative[3][:3, :3]
        R35 = R3.T @ r_ee
        # calculate theta 4 and theta 5
        theta4 = atan2(R35[1, 0], R35[0, 0])
        theta5 = atan2(R35[2, 0], R35[2, 1])

        curr_joint_vals = [theta1, theta2, theta3, theta4, theta5]
        #clip joints
        curr_joint_vals = np.clip(curr_joint_vals,
                                   [limit[0] for limit in self.joint_limits],
                                   [limit[1] for limit in self.joint_limits]
        )
        return [self.normalized_angle(theta) for theta in curr_joint_vals]
    
        # calculate angles
        #curr_joint_values = joint_values.copy()
        #beta = acos((self.l2**2 + self.l3**2 - (np.sqrt(self.l1**2 + self.l2**2))) / (2 * self.l2 * self.l3))
        #theta2 = pi-beta
        #alpha = atan(self.l2*sin(theta2), self.l1+self.l2*cos(theta2))
        #theta1 = atan(curr_joint_values[1], curr_joint_values[0]) - alpha
        
        # clip values to joint limits
        #theta1 = np.clip(theta1, self.joint_limits[0][0], self.joint_limits[0][1])
        #theta2 = np.clip(theta2, self.joint_limits[1][0], self.joint_limits[1][1])

        #return theta1, theta2

    def calc_numerical_ik(self, ee, joint_values: list, radians=True, tol = 0.005, limit = 200):
        # end effector positions
        curr_joints = np.array(joint_values, dtype=float)
        # 150 starting configs
        for _ in range (150):
            # run 200 iterations 
            for i in range(limit):
                ee_curr, _ = self.calc_forward_kinematics(curr_joints)
                # calculate the error between the current ee position and the target ee position
                dx = np.array([
                            ee.x - ee_curr.x,
                            ee.y - ee_curr.y,
                            ee.z - ee_curr.z,
                            ])
                # return the angle if the error is under tolerance
                if np.linalg.norm(dx) < tol:
                        return curr_joints
                # update the inverse jacobian
                #step = 0.1    
                curr_joints += self.inverse_jacobian(curr_joints) @ dx
                # clip joint values
                for i, (low, high) in enumerate(self.joint_limits):
                    curr_joints[i] = np.clip(curr_joints[i], low, high)
            # return random configuration if no solution is found within the limit
            curr_joints = np.array(ut.sample_valid_joints(self), dtype = float)
        return  np.zeros(len(curr_joints))

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """
        new_joint_values = list(joint_values[:self.num_dof])

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
       
        # Calculate joint velocities using the inverse Jacobian
        vel = vel[:3]  # Consider only the first two components of the velocity
        joint_vel = self.inverse_jacobian(new_joint_values) @ vel
       
        joint_vel = np.clip(joint_vel,
                            [limit[0] for limit in self.joint_vel_limits],
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values,
                               [limit[0] for limit in self.joint_limits],
                               [limit[1] for limit in self.joint_limits]
        )
       
        return new_joint_values

    def normalized_angle(self, angle):
        return(angle + pi) % (2 * pi) - pi  
   

    def jacobian(self, joint_values: list):
        """
        Calculates the 6xN Jacobian matrix for any number of joints.
       
        H_cumulative: List of (N+1) matrices [H0_0, H0_1, ..., H0_n]
        joint_types: List of strings ['R', 'R', 'P', ...] where R=Revolute, P=Prismatic.
                    If None, assumes all are Revolute.
        """
        _, h_list = self.calc_forward_kinematics(joint_values)

        H_cumulative = [np.eye(4)]
        for h in h_list:
            H_cumulative.append(H_cumulative[-1] @ h)
        num_joints = len(h_list)
           
        d_n = H_cumulative[-1][:3, 3] # end-effector position
        jacobian = np.zeros((6, num_joints))
       
        for i in range(num_joints):
            H_prev = H_cumulative[i]
            z_prev = H_prev[:3, 2] # z-axis of the previous frame
            d_prev = H_prev[:3, 3] # origin of the previous frame
           
            jv = np.cross(z_prev, (d_n - d_prev))
            jw = z_prev

            jacobian[:3,i] = jv # linear velocity
            jacobian[3:,i] = jw # angular velocity

        return jacobian[:3,:]

   

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        jacobian = self.jacobian(joint_values)
        lamda = 0.01
        damped_jacobian = jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T + lamda**2 * np.eye(jacobian.shape[0]))
        return damped_jacobian


if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = FiveDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()