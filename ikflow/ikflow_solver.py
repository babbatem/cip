from typing import List, Tuple, Optional, Union
import pickle
from time import time

from ikflow import config
from ikflow.supporting_types import IkflowModelParameters
from ikflow.robot_models import RobotModel
from ikflow.model import glow_cNF_model


import torch
import numpy as np


def draw_latent_noise(
    user_specified_latent_noise: Optional[torch.Tensor],
    latent_noise_distribution: str,
    latent_noise_scale: float,
    shape: Tuple[int, int],
):
    """Draw a sample from the latent noise distribution for running inference

    Args:
        user_specified_latent_noise (Optional[torch.Tensor]): _description_
        latent_noise_distribution (str): _description_
        latent_noise_scale (float): _description_
        shape (Tuple[int, int]): _description_

    Returns:
        _type_: _description_
    """
    assert latent_noise_distribution in ["gaussian", "uniform"]
    assert latent_noise_scale > 0
    assert len(shape) == 2
    if user_specified_latent_noise is not None:
        return user_specified_latent_noise
    if latent_noise_distribution == "gaussian":
        return latent_noise_scale * torch.randn(shape).to(config.device)
    elif latent_noise_distribution == "uniform":
        return 2 * latent_noise_scale * torch.rand(shape).to(config.device) - latent_noise_scale


class IkflowSolver:
    def __init__(self, hyper_parameters: IkflowModelParameters, robot_model: RobotModel):
        """Initialize an IkflowSolver."""
        assert isinstance(hyper_parameters, IkflowModelParameters)

        self.dim_cond = 7  # [x, y, z, q0, q1, q2, q3]
        if hyper_parameters.softflow_enabled:
            self.dim_cond = 8  # [x, ... q3, softflow_scale]   (softflow_scale should be 0 for inference)
        self.network_width = hyper_parameters.dim_latent_space
        self.nn_model = glow_cNF_model(hyper_parameters, robot_model, self.dim_cond, self.network_width)
        self.ndofs = robot_model.ndofs
        self.robot_model = robot_model

    def refine_solutions(
        self,
        ikflow_solutions: torch.Tensor,
        target_pose: Union[List[float], np.ndarray],
        positional_tolerance: float = 1e-3,
    ) -> Tuple[torch.Tensor, float]:
        """Refine a batch of IK solutions using the klampt IK solver

        Args:
            ikflow_solutions (torch.Tensor): A batch of IK solutions of the form [batch x ndofs]
            target_pose (Union[List[float], np.ndarray]): The target endpose(s). Must either be of the form
                                                            [x, y, z, q0, q1, q2, q3] or be a [batch x 7] numpy array
        Returns:
            torch.Tensor: A batch of IK refined solutions [batch x ndofs]
        """
        t0 = time()
        b = ikflow_solutions.shape[0]
        if isinstance(target_pose, list):
            target_pose = np.array(target_pose)
        if isinstance(target_pose, np.ndarray) and len(target_pose.shape) == 2:
            assert target_pose.shape[0] == b, f"target_pose.shape ({target_pose.shape[0]}) != [{b} x {self.ndofs}]"

        ikflow_solutions_np = ikflow_solutions.detach().cpu().numpy()
        refined = ikflow_solutions_np.copy()
        is_single_pose = (len(target_pose.shape) == 1) or (target_pose.shape[0] == 1)
        pose = target_pose

        for i in range(b):
            if not is_single_pose:
                pose = target_pose[i]
            ik_sol = self.robot_model.inverse_kinematics_klampt(
                pose, seed=ikflow_solutions_np[i], positional_tolerance=positional_tolerance
            )
            if ik_sol is not None:
                refined[i] = ik_sol

        return torch.from_numpy(refined).to(config.device), time() - t0

    def make_samples(
        self,
        y: List[float],
        m: int,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
        refine_solutions: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Run the network in reverse to generate samples conditioned on a pose y

        Args:
            y (List[float]): Target endpose of the form [x, y, z, q0, q1, q2, q3]
            m (int): The number of samples to draw
            latent_noise (Optional[torch.Tensor], optional): A batch of [batch x network_widthal] latent noise vectors.
                                                                Note that `y` and `m` will be ignored if this variable
                                                                is passed. Defaults to None.
            latent_noise_distribution (str): One of ["gaussian", "uniform"]
            latent_noise_scale (float, optional): The scaling factor for the latent noise samples. Samples from the
                                                    gaussian latent space are multiplied by this value.
        Returns:
            Tuple[torch.Tensor, float]:
                - A [batch x ndofs] batch of IK solutions
                - The runtime of the operation
        """
        assert len(y) == 7
        assert latent_noise_distribution in ["gaussian", "uniform"]
        t0 = time()

        # (:, 0:3) is x, y, z, (:, 3:7) is quat
        conditional = torch.zeros(m, self.dim_cond)
        conditional[:, 0:3] = torch.FloatTensor(y[:3])
        conditional[:, 3 : 3 + 4] = torch.FloatTensor(np.array([y[3:]]))
        conditional = conditional.to(config.device)

        latent_noise = draw_latent_noise(
            latent_noise, latent_noise_distribution, latent_noise_scale, (m, self.network_width)
        )
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.network_width
        output_rev, _ = self.nn_model(latent_noise, c=conditional, rev=True)
        solutions = output_rev[:, 0 : self.ndofs]
        runtime = time() - t0
        if not refine_solutions:
            return solutions, runtime
        refined, refinement_runtime = self.refine_solutions(solutions, y)
        return refined, runtime + refinement_runtime

    def make_samples_mult_y(
        self,
        ys: np.array,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
        refine_solutions: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Same as make_samples, but for multiple ys.

        ys: [batch x 7]
        """
        assert ys.shape[1] == 7
        t0 = time()
        m = ys.shape[0]

        # Note: No code change required here to handle using/not using softflow.
        conditional = torch.zeros(m, self.dim_cond)
        conditional[:, 0:7] = torch.FloatTensor(ys)
        conditional = conditional.to(config.device)
        latent_noise = draw_latent_noise(
            latent_noise, latent_noise_distribution, latent_noise_scale, (m, self.network_width)
        )
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.network_width
        output_rev, _ = self.nn_model(latent_noise, c=conditional, rev=True)
        solutions = output_rev[:, 0 : self.ndofs]
        runtime = time() - t0
        if not refine_solutions:
            return solutions, runtime
        refined, refinement_runtime = self.refine_solutions(solutions, ys)
        return refined, runtime + refinement_runtime

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""
        with open(state_dict_filename, "rb") as f:
            self.nn_model.load_state_dict(pickle.load(f))
