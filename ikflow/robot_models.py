from typing import List, Tuple, Optional

from ikflow.math_utils import geodesic_distance_between_quaternions

import klampt
from klampt.model import ik
from klampt.robotsim import RobotModel, RobotModelLink, IKSolver
from klampt.math import so3
import kinpy as kp
import numpy as np

# TODO(@jstmn): Delete contents of this file and use the `Jkinpylib` library once it's ready

# -----------------
# Testing functions

MAX_ALLOWABLE_L2_ERR = 5e-4
MAX_ALLOWABLE_ANG_ERR = 0.0008726646  # .05 degrees


def assert_endpose_position_almost_equal(endpoints1: np.array, endpoints2: np.array):
    """Check that the position of each pose is nearly the same"""
    l2_errors = np.linalg.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], axis=1)
    for i in range(l2_errors.shape[0]):
        assert l2_errors[i] < MAX_ALLOWABLE_L2_ERR


def assert_endpose_rotation_almost_equal(endpoints1: np.array, endpoints2: np.array):
    """Check that the rotation of each pose is nearly the same"""
    rotational_errors = geodesic_distance_between_quaternions(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
    for i in range(rotational_errors.shape[0]):
        assert (
            rotational_errors[i] < MAX_ALLOWABLE_ANG_ERR
        ), f"Rotation error {rotational_errors[i]} > {MAX_ALLOWABLE_ANG_ERR}"


# Gripe: Why are links actuated in the `Klampt/bin/RobotPose` tool? This is confusing and misleading
class RobotModel:
    def __init__(
        self,
        robot_name: str,
        urdf_filepath: str,
        joint_chain: List[str],
        actuated_joints: List[str],
        actuated_joints_limits: List[Tuple[float, float]],
        ndofs: int,
        end_effector_link_name: str,
        base_link_name: str,
    ):
        assert len(end_effector_link_name) > 0, f"End effector link name '{end_effector_link_name}' is empty"
        assert len(actuated_joints) == len(actuated_joints_limits)
        assert len(actuated_joints) == ndofs
        assert len(actuated_joints) <= len(joint_chain)
        for joint in actuated_joints:
            assert joint in joint_chain

        self._robot_name = robot_name
        self._ndofs = ndofs
        self._end_effector_link_name = end_effector_link_name
        self._actuated_joints = actuated_joints
        self._actuated_joints_limits = actuated_joints_limits

        # Initialize klampt
        # Note: Need to save `_klampt_wm` as a member variable otherwise you'll be doomed to get a segfault
        self._klampt_wm = klampt.WorldModel()
        assert self._klampt_wm.loadRobot(urdf_filepath), f"Error loading urdf '{urdf_filepath}'"
        self._klampt_robot: RobotModel = self._klampt_wm.robot(0)
        self._klampt_ee_link: RobotModelLink = self._klampt_robot.link(self._end_effector_link_name)
        self._klampt_config_dim = len(self._klampt_robot.getConfig())
        self._klampt_active_dofs = self._get_klampt_active_dofs()

        assert (
            self._klampt_robot.numDrivers() == ndofs
        ), f"# of active joints in urdf {self._klampt_robot.numDrivers()} doesn't equal `ndofs`: {self._ndofs}"

        # Initialize Kinpy.
        # Note: we use both klampt and kinpy to gain confidence that our FK function is correct. This
        # is a temporary measure until we are sure they are correct.
        with open(urdf_filepath) as f:
            self._kinpy_fk_chain = kp.build_chain_from_urdf(f.read().encode("utf-8"))
        self.assert_forward_kinematics_functions_equal()

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def ndofs(self) -> int:
        """Returns the number of degrees of freedom of the robot"""
        return self._ndofs

    @property
    def actuated_joints_limits(self) -> List[Tuple[float, float]]:
        return self._actuated_joints_limits

    def _get_klampt_active_dofs(self) -> List[int]:
        active_dofs = []
        self._klampt_robot.setConfig([0] * self._klampt_config_dim)
        idxs = [1000 * (i + 1) for i in range(self.ndofs)]
        q_temp = self._klampt_robot.configFromDrivers(idxs)
        for idx, v in enumerate(q_temp):
            if v in idxs:
                active_dofs.append(idx)
        assert len(active_dofs) == self.ndofs, f"len(active_dofs): {len(active_dofs)} != self.ndofs: {self.ndofs}"
        return active_dofs

    def sample(self, n: int, solver=None) -> np.ndarray:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors

        Args:
            n (int): _description_
            solver (_type_, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        angs = np.random.rand(n, self._ndofs)  # between [0, 1)
        joint_limits = [[l, u] for (l, u) in self._actuated_joints_limits]

        # Sample
        for i in range(self._ndofs):
            range_ = joint_limits[i][1] - joint_limits[i][0]
            assert range_ > 0
            angs[:, i] *= range_
            angs[:, i] += joint_limits[i][0]
        return angs

    def _x_to_qs(self, x: np.ndarray) -> List[List[float]]:
        """Return a list of klampt configurations (qs) from an array of joint angles (x)

        Args:
            x: (n x ndofs) array of joint angle settings

        Returns:
            A list of configurations representing the robots state in klampt
        """
        assert len(x.shape) == 2, f"x must be [m, ndof] (currently: {x.shape})"
        assert x.shape[1] == self._ndofs

        n = x.shape[0]
        qs = []
        for i in range(n):
            qs.append(self._klampt_robot.configFromDrivers(x[i].tolist()))
        return qs

    def _qs_to_x(self, qs: List[List[float]]) -> np.array:
        """Calculate joint angle values (x) from klampt configurations (qs)"""
        res = np.zeros((len(qs), self.ndofs))
        for idx, q in enumerate(qs):
            drivers = self._klampt_robot.configToDrivers(q)
            res[idx, :] = drivers
        return res

    def assert_forward_kinematics_functions_equal(self):
        """Test that kinpy and klampt the same poses"""
        n_samples = 10
        samples = self.sample(n_samples)
        kinpy_fk = self.forward_kinematics_kinpy(samples)
        klampt_fk = self.forward_kinematics_klampt(samples)
        assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)
        print(f"Success - klampt, kinpy forward-kinematics functions agree for '{self.robot_name}'")

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Forward Kinematics                                             ---
    # ---                                                                                                            ---

    def forward_kinematics_kinpy(self, x: np.ndarray) -> np.array:
        """Calculate forward kinematics using the kinpy library"""
        assert len(x.shape) == 2, f"x must be [m, ndof] (currently: {x.shape})"
        assert x.shape[1] == self._ndofs, f"x must be [m, {self._ndofs}] (currently: {x.shape[0]} x {x.shape[1]})"

        n = x.shape[0]
        y = np.zeros((n, 7))
        zero_transform = kp.transform.Transform()
        fk_dict = {}
        for joint_name in self._actuated_joints:
            fk_dict[joint_name] = 0.0

        def get_fk_dict(xs):
            for i in range(self.ndofs):
                fk_dict[self._actuated_joints[i]] = xs[i]
            return fk_dict

        for i in range(n):
            th = get_fk_dict(x[i])
            transform = self._kinpy_fk_chain.forward_kinematics(th, world=zero_transform)[self._end_effector_link_name]
            y[i, 0:3] = transform.pos
            y[i, 3:] = transform.rot
        return y

    def forward_kinematics_klampt(self, x: np.ndarray) -> np.array:
        """
        Returns the pose of the end effector for each joint parameter setting in x. Forward kinematics calculated with
        klampt
        """
        robot_configs = self._x_to_qs(x)

        dim_y = 7
        n = len(robot_configs)
        y = np.zeros((n, dim_y))

        for i in range(n):
            q = robot_configs[i]
            self._klampt_robot.setConfig(q)
            R, t = self._klampt_ee_link.getTransform()
            y[i, 0:3] = np.array(t)
            y[i, 3:] = np.array(so3.quaternion(R))
        return y

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Inverse Kinematics                                             ---
    # ---                                                                                                            ---

    def inverse_kinematics_klampt(
        self,
        pose: np.array,
        seed: Optional[np.ndarray] = None,
        positional_tolerance: float = 1e-3,
        n_tries: int = 50,
        verbosity: int = 0,
    ) -> Optional[np.array]:
        """Run klampts inverse kinematics solver with the given pose

        Note: If the solver fails to find a solution with the provided seed, it will rerun with a random seed

        Per http://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-IK.html#ik-solver:
        'To use the solver properly, you must understand how the solver uses the RobotModel:
            First, the current configuration of the robot is the seed configuration to the solver.
            Second, the robot's joint limits are used as the defaults.
            Third, the solved configuration is stored in the RobotModel's current configuration.'

        Args:
            pose (np.array): The target pose to solve for
            seed (Optional[np.ndarray], optional): A seed to initialize the optimization with. Defaults to None.
            verbosity (int): Set the verbosity of the function. 0: only fatal errors are printed. Defaults to 0.
        """
        assert len(pose.shape) == 1
        assert pose.size == 7
        if seed is not None:
            assert isinstance(seed, np.ndarray), f"seed must be a numpy array (currently {type(seed)})"
            assert len(seed.shape) == 1, f"Seed must be a 1D array (currently: {seed.shape})"
            assert seed.size == self.ndofs
            seed_q = self._x_to_qs(seed.reshape((1, self.ndofs)))[0]

        max_iterations = 150
        R = so3.from_quaternion(pose[3 : 3 + 4])
        obj = ik.objective(self._klampt_ee_link, t=pose[0:3].tolist(), R=R)

        for _ in range(n_tries):
            solver = IKSolver(self._klampt_robot)
            solver.add(obj)
            solver.setActiveDofs(self._klampt_active_dofs)
            solver.setMaxIters(max_iterations)
            # TODO(@jstmn): What does 'tolarance' mean for klampt? Positional error? Positional error + orientation error?
            solver.setTolerance(positional_tolerance)

            # `sampleInitial()` needs to come after `setActiveDofs()`, otherwise x,y,z,r,p,y of the robot will
            # be randomly set aswell <(*<_*)>
            if seed is None:
                solver.sampleInitial()
            else:
                # solver.setBiasConfig(seed_q)
                self._klampt_robot.setConfig(seed_q)

            res = solver.solve()
            if not res:
                if verbosity > 0:
                    print(
                        "  inverse_kinematics_klampt() IK failed after",
                        solver.lastSolveIters(),
                        "optimization steps, retrying (non fatal)",
                    )

                # Rerun the solver with a random seed
                if seed is not None:
                    return self.inverse_kinematics_klampt(
                        pose, seed=None, positional_tolerance=positional_tolerance, verbosity=verbosity
                    )

                continue

            if verbosity > 1:
                print("Solved in", solver.lastSolveIters(), "iterations")
                residual = solver.getResidual()
                print("Residual:", residual, " - L2 error:", np.linalg.norm(residual[0:3]))

            return self._qs_to_x([self._klampt_robot.getConfig()])

        if verbosity > 0:
            print("inverse_kinematics_tracik() - Failed to find IK solution after", n_tries, "optimization attempts")
        return None


# ----------------------------------------------------------------------------------------------------------------------
# The robots
#


class PandaArm(RobotModel):
    name = "panda_arm"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [
            (-2.8973, 2.8973),  # panda_joint1
            (-1.7628, 1.7628),  # panda_joint2
            (-2.8973, 2.8973),  # panda_joint3
            (-3.0718, -0.0698),  # panda_joint4
            (-2.8973, 2.8973),  # panda_joint5
            (-0.0175, 3.7525),  # panda_joint6
            (-2.8973, 2.8973),  # panda_joint7
        ]
        actuated_joints = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        ndofs = 7
        joint_chain = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_joint8",
        ]

        urdf_filepath = "third_party/robosuite/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm.urdf"
        end_effector_link_name = "panda_link8"
        base_link_name = "panda_link0"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
            base_link_name,
        )


def get_robot(robot_name: str) -> RobotModel:
    if robot_name == "panda_arm":
        return PandaArm()
    assert False, f"robot '{robot_name}' not found"


""" Example usage

python ikflow/robot_models.py
"""

if __name__ == "__main__":
    # Smoke test to check that we can create a RobotModel for the panda robot
    robot = PandaArm()
