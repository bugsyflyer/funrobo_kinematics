import math
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate



class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate forward kinematics based on the provided joint angles.
        
        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        curr_joint_values = joint_values.copy()
        
        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]
        
        # Ensure that the joint angles respect the joint limits
        #for i, theta in enumerate(curr_joint_values):
            #curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])

        # Set the Denavit-Hartenberg parameters for each joint
        DH = np.zeros((self.num_dof, 4)) # [theta, d, a, alpha]
        DH[0] = [curr_joint_values[0], self.l1, 0, -np.pi/2]
        DH[1] = [curr_joint_values[1] - np.pi/2, 0, self.l2, np.pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, np.pi]
        DH[3] = [curr_joint_values[3] + np.pi/2, 0, 0, np.pi/2]
        DH[4] = [curr_joint_values[4], self.l4 + self.l5, 0, 0]

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
    

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy, vz].
        """
        new_joint_values = joint_values.copy()

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate the joint velocity using the inverse Jacobian
        # joint_vel = self.inverse_jacobian(new_joint_values, pseudo=True) @ vel
        joint_vel = self.damped_inverse_jacobian(new_joint_values) @ vel

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
    

    def jacobian(self, joint_values: list):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            Jacobian matrix (3x5).
        """
        _, Hlist = self.calc_forward_kinematics(joint_values)

        # Precompute transformation matrices for efficiency
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            H_curr = H_cumulative[i]
            H_final = H_cumulative[-1]
            
            # Calculate position vector r
            r = (H_final @ O0 - H_curr @ O0)[:3]

            # Compute the rotation axis z
            z = H_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        return ut.near_zero(jacobian)
  

    def inverse_jacobian(self, joint_values: list, pseudo=True):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.
        
        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).
        
        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = self.jacobian(joint_values)

        if pseudo:
            return np.linalg.pinv(self.jacobian3x5(joint_values))
        else:
            return np.linalg.inv(self.jacobian3x5(joint_values))
        
        
    def damped_inverse_jacobian(self, joint_values: list, damping_factor=0.025):
        
        J = self.jacobian(joint_values)
        JT = np.transpose(J)
        I = np.eye(3)
        return JT @ np.linalg.inv(J @ JT + (damping_factor**2)*I)
    
    def calc_inverse_kinematics(self, ee, init_joint_values, soln=0):
        #currently does not account for multiple solns
        d5 = self.l4 + self.l5
        H5 = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))
        w_x = ee.x - d5*H5[0,2]
        w_y = ee.y - d5*H5[1,2]
        w_z = ee.z - d5*H5[2,2]

        theta1list = []
        theta1list.append(math.atan2(ee.y, ee.x)) #forward
        
        if math.atan2(ee.y, ee.x) > 0:
            theta1list.append(math.atan2(ee.y, ee.x) - np.pi)
        else:
            theta1list.append(math.atan2(ee.y, ee.x) + np.pi)
        
        r = math.sqrt(w_x**2 + w_y**2)
        s = w_z - self.l1
        L = math.sqrt(r**2 + s**2)
        
        cos_theta_3 = (self.l2**2 + self.l3**2 - L**2)/(2*self.l2*self.l3)
        cos_theta_3 = np.clip(cos_theta_3, -1.0, 1.0)
        
        theta3list = []
        theta3list.append(np.pi - math.acos(cos_theta_3)) #elbow up
        theta3list.append(-(np.pi - math.acos(cos_theta_3))) #elbow down
        
        #need backward theta 2 to account for backward theta1, so should have 4 sols total
        theta2list = []
        # theta2list.append(math.atan2(s,r) - math.atan2(self.l3*math.sin(theta3list[0]), self.l2 + self.l3*math.cos(theta3list[0])))
        # theta2list.append(math.atan2(s,r) - math.atan2(self.l3*math.sin(theta3list[1]), self.l2 + self.l3*math.cos(theta3list[1])))
        g1 = self.l3 * math.sin(theta3list[0])
        b1 = math.asin(g1/L)
        g2 = self.l3 * math.sin(theta3list[1])
        b2 = math.asin(g2/L)
        theta2list.append(-(math.atan2(s,r) - b1 - np.pi/2))
        theta2list.append((math.atan2(s,r) + b2 - np.pi/2))
        
        possible_joint_values = []
        for i in range(2):
            for j in range(2):
                # c23 = math.cos(theta2list[j])*math.cos(theta3list[j]) - math.sin(theta2list[j])*math.sin(theta3list[j])
                # s23 = math.sin(theta2list[j])*math.cos(theta3list[j]) + math.cos(theta2list[j])*math.sin(theta3list[j])
                # R0_3 = np.array([[math.cos(theta1list[i])*math.cos(s23), math.cos(theta1list[i])*math.cos(c23), math.sin(theta1list[i])],
                #                 [math.sin(theta1list[i])*math.sin(s23), math.sin(theta1list[i])*math.sin(s23), -math.cos(theta1list[i])],
                #                 [math.cos(c23), -math.sin(s23), 0]])
                # R3_5 = R0_3.T @ H5
                # R5_3 = R3_5.T
                dh1 = ut.dh_to_matrix([theta1list[i], self.l1, 0, np.pi/2])
                dh2 = ut.dh_to_matrix([theta2list[j]+np.pi/2, 0, self.l2, 0])
                dh3 = ut.dh_to_matrix([-theta3list[j], 0, self.l3, 0])
                
                t03 = dh1@dh2@dh3
                r03 = t03[:3,:3]
                r35 = np.transpose(r03)@H5
    
                theta4 = math.atan2(r35[1,2], r35[0,2])
                theta5 = math.atan2(r35[2, 0], r35[2,1])
                possible_joint_values.append([theta1list[i], theta2list[j], theta3list[j], theta4, theta5])
        
        last_error = 1000
        best_joints_index = 0
        last_best_joints_index = 0
        for i in range(len(possible_joint_values)):
            [ee_guess, _] = self.calc_forward_kinematics(possible_joint_values[i])
            print(f"joint vals: {possible_joint_values[i]}")
            error = abs(ee_guess.x - ee.x) + abs(ee_guess.y - ee.y) + abs(ee_guess.z - ee.z)
            print(f"error: {error}")
            if error < last_error:
                last_best_joints_index = best_joints_index
                best_joints_index = i
            last_error = error
        
        if soln == 0:      
            return possible_joint_values[best_joints_index]
        return possible_joint_values[last_best_joints_index]
        

    
    def calc_numerical_ik(self,ee,init_joint_values,tol: float = 0.002,ilimit: int = 200):
        error = [100, 100, 100]
    
        while np.linalg.norm(error) > tol:
            joint_values = np.array([np.random.uniform(low, high) for low, high in self.joint_limits])
            #print(f"joint vals: {joint_values}")
            for _ in range(ilimit):
                [ee_guess, _] = self.calc_forward_kinematics(joint_values)
                error = np.array([
                    ee_guess.x - ee.x,
                    ee_guess.y - ee.y,
                    ee_guess.z - ee.z,
                ])

                #if converged
                if np.linalg.norm(error) < tol:
                    print(f"error: {error}")
                    return joint_values

                #damped inv jacobian
                J = self.damped_inverse_jacobian(joint_values)

                joint_values = joint_values + (J @ error)


if __name__ == "__main__":
    
    model = FiveDOFRobot()
    
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()