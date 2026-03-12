from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)

class FiveDOF(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
        #self.joint_limits = [[-pi, pi] for _ in range(5)]

    def _compute_transforms(self, joint_values):
        """
        Helper to calculate cumulative transformation matrices (H_cumulative)
        and individual joint transforms (Hlist) based on the FiveDOF robot model.
       
        Args:
            joint_values (list/array): Processed joint angles in radians.
        """
        theta = joint_values
        DH = np.array([ # DH parameters for each joint
            [theta[0],          self.l1,         0,       -pi/2],
            [theta[1]-(pi/2),   0,               self.l2,  pi],
            [theta[2],          0,               self.l3,  pi],
            [theta[3]+(pi/2),   0,               0,        pi/2],
            [theta[4],          self.l4+self.l5, 0,        0]
        ])

        Hlist = [ut.dh_to_matrix(dh) for dh in DH] # Compute transformation matrices for each joint

        # Compute cumulative transformations
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
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
       
        # DH parameters for each joint
        DH = np.zeros((self.num_dof, 4))
        DH[0] = [curr_joint_values[0], self.l1, 0, -pi/2]
        DH[1] = [curr_joint_values[1]-(pi/2), 0, self.l2, pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, pi]
        DH[3] = [curr_joint_values[3]+(pi/2), 0, 0, pi/2]
        DH[4] = [curr_joint_values[4], self.l4+self.l5, 0, 0]

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
   

    def calc_numerical_ik(self, ee, joint_values, tol=0.002, ilimit=100):
        """
        Numerical IK with angles wrapped to [-pi, pi] and joint limit enforcement.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles (radians).
            tol (float, optional): Convergence tolerance. Defaults to 0.002.
            ilimit (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            list[float]: Estimated joint angles in radians, wrapped to [-pi, pi] and within joint limits.
        """
        # unpack end effector position and rotation
        x_target, y_target, z_target = ee.x, ee.y, ee.z
        new_joint_values = np.array(joint_values, dtype=float)

        for _ in range(100):  # allow 100 attempted starting configurations
            for _ in range(ilimit): # 100 iterations for each attempted configuration
                # get the end effector position based on the current joint guess and find the error from the desired position
                current_ee, _ = self.calc_forward_kinematics(new_joint_values)
                error = np.array([x_target, y_target, z_target]) - np.array([current_ee.x, current_ee.y, current_ee.z])

                # if the error is within tolerance, return the joint angle solution
                if np.linalg.norm(error) <= tol:
                    return new_joint_values

                # get next iteration by updating with inverse jacobian
                new_joint_values += self.inverse_jacobian(new_joint_values) @ error

                # enforce joint limits
                for i, (low, high) in enumerate(self.joint_limits):
                    new_joint_values[i] = np.clip(new_joint_values[i], low, high)

            # if not converged, return a random configuration and try again
            new_joint_values = np.array(ut.sample_valid_joints(self), dtype=float)
       
        # return null if not converged
        return np.zeros(len(joint_values))


    def calc_inverse_kinematics(self, ee, joint_values=None, soln=0):
        """
        Calculates the analytical inverse position kinematics for the 5-DOF manipulator.
        Utilizes kinematic decoupling to resolve the proximal spatial positioning
        independently from the distal spatial orientation.
        """
        # Extract end effector position and orientation
        p_ee = np.array([ee.x, ee.y, ee.z])
        R_05 = ut.euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
       
        # Calculate wrist position via end effector position and orientation
        d_6 = self.l4 + self.l5
        p_wrist = p_ee - d_6 * (R_05 @ np.array([0, 0, 1]))
       
        wx, wy, wz = p_wrist[0], p_wrist[1], p_wrist[2]
       
        # Calculate all possible solutions
        solutions = []
       
        # Begin joint solutions
        # First branch is: Base Yaw (Front vs Back)
        theta_1_opts = [atan2(wy, wx), atan2(-wy, -wx)]
       
        for theta_1 in theta_1_opts:
            # r is distance in the plane defined by theta_1
            r = wx * cos(theta_1) + wy * sin(theta_1)
            s = wz - self.l1
            # Calculate square of the distance between the wrist and the shoulder
            L_sq = r**2 + s**2
           
            # Define link lengths (proximal = sholder to elbow, distal = elbow to wrist)
            l_proximal = self.l2
            l_distal = self.l3
           
            # Law of Cosines
            numerator = l_proximal**2 + l_distal**2 - L_sq
            denominator = 2 * l_proximal * l_distal
           
            # Check if target is reachable
            if abs(numerator) > abs(denominator):
                continue
               
            cos_beta = numerator / denominator
            beta = np.arccos(cos_beta)
           
            # Second branch is: Elbow Config (Up vs Down)
            for theta_3_candidate in [np.pi - beta, -(np.pi - beta)]:
                theta_3 = theta_3_candidate
               
                # alpha
                alpha = np.arctan2(l_distal * np.sin(theta_3), l_proximal + l_distal * np.cos(theta_3))

                # gamma
                gamma = np.arctan2(s, r)

                # theta_2
                theta_2 = (np.pi / 2) - (gamma - alpha)
               
                # Solve for Wrist (theta_4, theta_5)
                q_list = [theta_1, theta_2, theta_3, 0, 0]
               
                H_cumulative, _ = self._compute_transforms(q_list)
                R_03 = H_cumulative[3][:3, :3]
               
                R_35 = R_03.T @ R_05
               
                # Extract from matrix (see derivation)
                theta_4 = atan2(R_35[1, 2], R_35[0, 2])
                theta_5 = atan2(R_35[2, 0], R_35[2, 1])
               
                candidate_q = [theta_1, theta_2, theta_3, theta_4, theta_5]
                candidate_q = [self.normalize_angle(q) for q in candidate_q]
               
                # Check limits
                if ut.check_joint_limits(candidate_q, self.joint_limits):
                    solutions.append(candidate_q)

        # Sort solutions by error
        def get_error(q):
            fk_ee, _ = self.calc_forward_kinematics(q)
            return np.linalg.norm(np.array([ee.x, ee.y, ee.z]) - np.array([fk_ee.x, fk_ee.y, fk_ee.z]))
       
        solutions.sort(key=get_error)

        if not solutions:
            return [0, 0, 0, 0, 0] # Return default position if no valid solutions are found

        # If a specific solution index is requested, return it if valid index
        if soln < len(solutions):
            return solutions[soln]

        # Return the best solution by default
        return solutions[0]

   
    def normalize_angle(self, angle):
        """Normalize an angle to the range (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the robot.

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
       
        curr_joint_values = joint_values.copy()

        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
       
        # DH parameters for each joint
        DH = np.zeros((self.num_dof, 4))
        DH[0] = [curr_joint_values[0], self.l1, 0, -pi/2]
        DH[1] = [curr_joint_values[1]-(pi/2), 0, self.l2, pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, pi]
        DH[3] = [curr_joint_values[3]+(pi/2), 0, 0, pi/2]
        DH[4] = [curr_joint_values[4], self.l4+self.l5, 0, 0]

        # Compute the transformation matrices
        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        p_ee = H_cumulative[-1][:3, 3]
        J = np.zeros((6, self.num_dof))
       
        for i in range(self.num_dof):
            transform_i = H_cumulative[i]
            z_axis = transform_i[:3, 2]
            p_joint = transform_i[:3, 3]
            J[:3, i] = np.cross(z_axis, (p_ee - p_joint))
            J[3:, i] = z_axis
        return J[:3, :]
   

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.jacobian(joint_values))
   
if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = FiveDOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()

