from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class TwoDOFRobot(TwoDOFRobotTemplate):
    def __init__(self):
        super().__init__()


    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        
        th1, th2 = curr_joint_values[0], curr_joint_values[1]
        l1, l2 = self.l1, self.l2

        H0_1 = np.array([[cos(th1), -sin(th1), 0, l1*cos(th1)],
                         [sin(th1), cos(th1), 0, l1*sin(th1)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
                        )

        H1_2 = np.array([[cos(th2), -sin(th2), 0, l2*cos(th2)],
                         [sin(th2), cos(th2), 0, l2*sin(th2)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
                        )
        
        Hlist = [H0_1, H1_2]

        # Calculate EE position and rotation
        H_ee = H0_1@H1_2  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist
    
    def calc_inverse_kinematics(self, ee):

        x, y = ee.x, ee.y
        l1, l2 = self.l1, self.l2

        r2 = x**2 + y**2

        #theta2
        c2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)

        #safety stuff
        c2 = np.clip(c2, -1.0, 1.0)

        s2 = sqrt(1 - c2**2)

        theta2_plus = atan2(s2, c2)     # elbow up
        theta2_minus = atan2(-s2, c2)   # elbow down

        #theta1
        def solve_theta1(theta2):
            return atan2(y, x) - atan2(
                l2*sin(theta2),
                l1 + l2*cos(theta2)
            )

        theta1_plus = solve_theta1(theta2_plus)
        theta1_minus = solve_theta1(theta2_minus)

        return [
            [theta1_plus, theta2_plus],
            [theta1_minus, theta2_minus]
        ]
    
    def calc_numerical_ik(
        self,
        ee,
        init_joint_values,
        tol: float = 1e-3,
        ilimit: int = 200):

        joint_values = np.array(init_joint_values, dtype=float)

        alpha = 0.3          # smaller step
        damping = 0.05       # DLS stability

        for _ in range(ilimit):

            ee_guess, _ = self.calc_forward_kinematics(joint_values)

            error = np.array([
                ee.x - ee_guess.x,
                ee.y - ee_guess.y
            ])

            # convergence check
            if np.linalg.norm(error) < tol:
                return joint_values

            #damped inv jacobian
            J = self.jacobian(joint_values)
            JT = J.T

            J_damped = JT @ np.linalg.inv(
                J @ JT + (damping**2) * np.eye(2)
            )

            dq = alpha * (J_damped @ error)

            joint_values = joint_values + dq

        return joint_values
        
    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the robot. 

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
        print(f"Joint values: {joint_values}")
        
        return np.array([
            [-self.l1 * sin(joint_values[0]) - self.l2 * sin(joint_values[0] + joint_values[1]), 
             -self.l2 * sin(joint_values[0] + joint_values[1])],
            [self.l1 * cos(joint_values[0]) + self.l2 * cos(joint_values[0] + joint_values[1]), 
             self.l2 * cos(joint_values[0] + joint_values[1])]
        ])
    

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        damping= 0.1
        J = self.jacobian(joint_values)

        JT = J.T
        lambda2_I = (damping**2) * np.eye(J.shape[0])

        return JT @ np.linalg.inv(J @ JT + lambda2_I)


if __name__ == "__main__":
    model = TwoDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()