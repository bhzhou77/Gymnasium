__credits__ = ["Kallinteris-Andreas", "Rushiv Arora", "Baohua Zhou"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

"""
'fl_hip': 0,
            'fl_uleg': 1,
            'fl_lleg': 2,
            'fl_foot': 3,
            'fr_hip': 4,
            'fr_uleg': 5,
            'fr_lleg': 6,
            'fr_foot': 7,
            'hl_hip': 8,
            'hl_uleg': 9,
            'hl_lleg': 10,
            'hl_foot': 11,
            'hr_hip': 12,
            'hr_uleg': 13,
            'hr_lleg': 14,
            'hr_foot': 15,

'fl_hx', 'fl_hy', 'fl_kn', 'fr_hx', 'fr_hy', 'fr_kn', 'freejoint', 'hl_hx', 'hl_hy', 'hl_kn', 'hr_hx', 'hr_hy', 'hr_kn'

[ 0.785398  2.29511  -0.254402  0.785398  2.24363  -0.255648  0.785398
  2.29511  -0.247067  0.785398  2.29511  -0.248282]
[-0.785398 -0.898845 -2.7929   -0.785398 -0.898845 -2.7929   -0.785398
 -0.898845 -2.7929   -0.785398 -0.898845 -2.7929  ]
"""


class SpotEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on robot dog Spot from Boston Dynamics.
    The Spot has 16 body parts and 12 joints connecting them (excluding the body piece in the center).
    The goal is to apply angle values to the joints to make the Spot perform various gaits, such as walk, trot and gallop.


    ## Action Space
    ```{figure} action_space_figures/spotanatomy.png
    :name: spotanatomy
    ```

    The action space is a `Box(low, high, (12,), float32)`. An action represents the angle values applied at the hinge joints.

    | Num | Action                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit) |
    | --- | ------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ----------- |
    | 0   | Angle applied on the front left hip rotor         | -0.785398   | 0.785398    | fl_hx                            | hinge | angle (rad) |
    | 1   | Angle applied on the front left upper leg rotor   | -0.898845   | 2.29511     | fl_hy                            | hinge | angle (rad) |
    | 2   | Angle applied on the front left lower leg rotor   | -2.7929     | -0.254402   | fl_kn                            | hinge | angle (rad) |
    | 3   | Angle applied on the front right hip rotor        | -0.785398   | 0.785398    | fr_hx                            | hinge | angle (rad) |
    | 4   | Angle applied on the front right upper leg rotor  | -0.898845   | 2.24363     | fr_hy                            | hinge | angle (rad) |
    | 5   | Angle applied on the front right lower leg rotor  | -2.7929     | -0.255648   | fr_kn                            | hinge | angle (rad) |
    | 6   | Angle applied on the hind left hip rotor          | -0.785398   | 0.785398    | hl_hx                            | hinge | angle (rad) |
    | 7   | Angle applied on the hind left upper leg rotor    | -0.898845   | 2.29511     | hl_hy                            | hinge | angle (rad) |
    | 8   | Angle applied on the hind left lower leg rotor    | -2.7929     | -0.247067   | hl_kn                            | hinge | angle (rad) |
    | 9   | Angle applied on the hind right hip rotor         | -0.785398   | 0.785398    | hr_hx                            | hinge | angle (rad) |
    | 10  | Angle applied on the hind right upper leg rotor   | -0.898845   | 2.29511     | hr_hy                            | hinge | angle (rad) |
    | 11  | Angle applied on the hind right lower leg rotor   | -2.7929     | -0.248282   | hr_kn                            | hinge | angle (rad) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (18 elements by default):* The first two are the y and z coordinates of the body, the next four are the quaternion of the free joints. The rest are the angular values of the robot's joints.
    - *qvel (18 elements):* The velocities of qpos (their derivatives).

    By default, the observation does not include the robot's x-coordinate.
    This can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (37,), float64)`, where the first observation element is the x-coordinate of the robot.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (36,), float64)` where the elements are as follows:


    TO BE DETERMINED.


    ## Rewards
    The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.

    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Spot moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the "tip" ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime` which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Spot for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{9}, reset\_noise\_scale \times I_{9}]}$.
    The initial velocity state is $\mathcal{N}(0_{9}, reset\_noise\_scale^2 \times I_{9})$.

    where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The Spot never terminates.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Spot provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Spot-v0', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default              | Description                                                                                                                                                                                         |
    | -------------------------------------------- | --------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"spot_scene.xml"`   | Path to a MuJoCo model                                                                                                                                                                              |
    | `forward_reward_weight`                      | **float** | `1`                  | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                            |
    | `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for _ctrl_cost_ weight (see `Rewards` section)                                                                                                                                               |
    | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                       |
    | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Restored the `xml_file` argument (was removed in `v4`).
        - Renamed `info["reward_run"]` to `info["reward_forward"]` to be consistent with the other environments.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "spot_scene.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }
