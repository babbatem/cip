from typing import Tuple, Union, Dict
import os
import random

from ikflow import config
from ikflow.robot_models import RobotModel, get_robot
from ikflow.ikflow_solver import IkflowSolver, IkflowModelParameters
from ikflow.math_utils import rotation_matrix_from_quaternion, geodesic_distance

import numpy as np
import torch


# _______________________
# Prediction errors


def get_solution_errors(
    robot_model: RobotModel, solutions: Union[torch.Tensor, np.ndarray], target_pose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the L2 and angular errors of calculated ik solutions for a given target_pose. Note: this function expects
    multiple solutions but only a single target_pose. All of the solutions are assumed to be for the given target_pose

    Args:
        robot_model (RobotModel): The RobotModel which contains the FK function we will use
        solutions (Union[torch.Tensor, np.ndarray]): [n x 7] IK solutions for the given target pose
        target_pose (np.ndarray): [7] the target pose the IK solutions were generated for

    Returns:
        Tuple[np.ndarray, np.ndarray]: The L2, and angular (rad) errors of IK solutions for the given target_pose
    """
    ee_pose_ikflow = robot_model.forward_kinematics_klampt(solutions[:, 0 : robot_model.ndofs].cpu().detach().numpy())

    # Positional Error
    l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_pose[0:3], axis=1)

    # Angular Error
    n_solutions = solutions.shape[0]
    rot_target = np.tile(target_pose[3:], (n_solutions, 1))
    rot_output = ee_pose_ikflow[:, 3:]

    output_R9 = rotation_matrix_from_quaternion(torch.Tensor(rot_output).to(config.device))
    target_R9 = rotation_matrix_from_quaternion(torch.Tensor(rot_target).to(config.device))
    ang_errors = geodesic_distance(target_R9, output_R9).cpu().data.numpy()
    assert l2_errors.shape == ang_errors.shape
    return l2_errors, ang_errors


# _______________________
# Model loading utilities


def get_ik_solver(
    model_weights_filepath: str, model_hyperparameters: Dict, robot_name: str
) -> Tuple[IkflowSolver, IkflowModelParameters, RobotModel]:
    """Build and return a `IkflowSolver` using the model weights saved in the file `model_weights_filepath` for the
    given robot and with the given hyperparameters

    Args:
        model_weights_filepath (str): The filepath for the model weights
        model_hyperparameters (Dict): The hyperparameters used for the NN
        model_hyperparameters (Dict): The hyperparameters used for the NN

    Returns:
        Tuple[IkflowSolver, IkflowModelParameters]:
            - An `IkflowSolver` IK solver
            - `IkflowModelParameters` the solvers hyperparameters
            - The corresponding robot
    """
    assert os.path.isfile(
        model_weights_filepath
    ), f"File '{model_weights_filepath}' was not found. Unable to load model weights"

    # Build GenerativeIKSolver and set weights
    hyper_parameters = IkflowModelParameters()
    hyper_parameters.__dict__.update(model_hyperparameters)
    robot_model = get_robot(robot_name)
    ik_solver = IkflowSolver(hyper_parameters, robot_model)
    ik_solver.load_state_dict(model_weights_filepath)
    return ik_solver, hyper_parameters, robot_model


# _____________
# Pytorch utils


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(0)
    print("set_seed() - random int: ", torch.randint(0, 1000, (1, 1)).item())
