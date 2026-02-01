# from scripts.demo import TwoDOFRobot
from solutions.demo_for_class import TwoDOFRobot, ScaraRobot
import funrobo_kinematics.core.utils as ut

from typing import List, Sequence

import pytest
import yaml

# -----------------------------------------------------------------------------
# Test configuration
# -----------------------------------------------------------------------------
# Choose which robot model + corresponding test data file to validate.

# robot_model = TwoDOFRobot()
# test_file = "tests/data/two_dof_fk_test_data.yaml"

robot_model = ScaraRobot()
test_file = 'tests/data/scara_fk_test_data.yaml'


joint_values_list = [ut.sample_valid_joints(robot_model) for _ in range(100)]
ee_list = []

for joint_values in joint_values_list:
    ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)
    ee_list.append([float(ee.x), float(ee.y)])

ids = [f"joint_values_{i}={[round(x,2) for x in q]} | position_{i}={[round(x,2) for x in ee_list[i]]}" for i, q in enumerate(joint_values_list)]


# -----------------------------------------------------------------------------
# Load test cases from YAML
# -----------------------------------------------------------------------------
with open(test_file, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

joint_values_list = data["joint_values"]
ee_list = data["ee"]


@pytest.mark.parametrize(
    "joint_values, ee",
    list(zip(joint_values_list, ee_list)),
    ids=[f"position_{i}={[round(x,2) for x in ee_list[i]]}" for i, q in enumerate(joint_values_list)]
)
def test_forward_kinematics(joint_values: Sequence[float], expected_ee: Sequence[float]) -> None:
    ee, _ = robot_model.calc_forward_kinematics(list(joint_values), radians=True)
    assert [ee.x, ee.y, ee.z] == pytest.approx(expected_ee, abs=1e-3)