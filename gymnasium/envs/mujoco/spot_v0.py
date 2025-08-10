__credits__ = ["Kallinteris-Andreas", "Rushiv Arora", "Baohua Zhou"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

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

    The action space is a `Box(low, high, (12,))`. An action represents the angle values applied at the hinge joints.

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

    - *qpos (18 elements by default):* The first two are the y and z coordinates of the body, the next four are the quaternion of the free joint. The rest are the angular values of the robot's joints.
    - *qvel (18 elements):* The velocities of qpos (their derivatives).

    By default, the observation does not include the robot's x-coordinate.
    This can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (37,))`, where the first observation element is the x-coordinate of the robot.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (36,))` where the elements are as follows:


    TO BE DETERMINED.


    ## Rewards
    The total reward is: ***reward*** *=* *tracking_lin_vel_reward - action_cost*.

    - *tracking_lin_vel_reward*:
    A reward for moving forward,
    this reward would be positive if the Spot moves in the xy plane.
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the "tip" ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $10$),
    and `frametime` which is $0.002$ - so the default is $dt = 10 \times 0.002 = 0.02$,
    $w_{forward}$ is the `tracking_lin_vel_reward_weight` (default is $1$).
    - *action_cost*:
    A negative reward to penalize the Spot for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `action_cost_weight` (default is $0.1$).

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
    env = gym.make('Spot-v0', action_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default              | Description                                                                                                                                                                                         |
    | -------------------------------------------- | --------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"spot_scene.xml"`   | Path to a MuJoCo model                                                                                                                                                                              |
    | `tracking_lin_vel_reward_weight`             | **float** | `1`                  | Weight for _tracking_lin_vel_reward_ term (see `Rewards` section)                                                                                                                                   |
    | `action_cost_weight`                         | **float** | `-0.1`               | Weight for _action_cost_ weight (see `Rewards` section)                                                                                                                                             |
    | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                       |
    | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |

    ## Version History
    * v0:
        - Adapted from half_cheetah_v5.
        - Changed the `.xml` file to the one for the spot.
        - Changed the frame_skip from 5 to 10.
        - Put minus signs the the weights for the costs.
        - Renamed `info["reward_forward"]` to `info["reward_tracking_lin_vel"]`.
        - Added more rewards and costs.
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
        xml_file: str = "spot_scene_v0.xml",
        frame_skip: int = 10,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        keep_upright_reward_weight: float = 10.0,
        tracking_lin_vel_reward_weight: float = 1.0,
        tracking_ang_vel_reward_weight: float = 1.0,
        upward_orientation_cost_weight: float = -10.0,
        lin_vel_z_cost_weight: float = -1.0,
        ang_vel_xy_cost_weight: float = -1.0,
        ang_vel_gyro_cost_weight: float = -0.0,
        action_cost_weight: float = -5.0,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            keep_upright_reward_weight,
            tracking_lin_vel_reward_weight,
            tracking_ang_vel_reward_weight,
            upward_orientation_cost_weight,
            lin_vel_z_cost_weight,
            ang_vel_xy_cost_weight,
            ang_vel_gyro_cost_weight,
            action_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._keep_upright_reward_weight = keep_upright_reward_weight
        self._tracking_lin_vel_reward_weight = tracking_lin_vel_reward_weight
        self._tracking_ang_vel_reward_weight = tracking_ang_vel_reward_weight
        self._upward_orientation_cost_weight = upward_orientation_cost_weight
        self._lin_vel_z_cost_weight = lin_vel_z_cost_weight
        self._ang_vel_xy_cost_weight = ang_vel_xy_cost_weight
        self._ang_vel_gyro_cost_weight = ang_vel_gyro_cost_weight
        self._action_cost_weight = action_cost_weight

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

    def step(self, action):
        xy_position_before = self.data.qpos[:2]
        z_position_before = self.data.qpos[2]
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[:2]
        z_position_after = self.data.qpos[2]

        tracking_lin_vel = (xy_position_after - xy_position_before) / self.dt
        lin_vel_z = (z_position_after - z_position_before) / self.dt

        ang_vel = self.get_global_angvel(self.data)
        ang_vel_gyro = self.get_gyro(self.data)
        gravity_vec = self.get_gravity(self.data)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(tracking_lin_vel, gravity_vec, lin_vel_z, ang_vel, ang_vel_gyro, action)
        info = {"xy_position": xy_position_after, 
                "z_position": z_position_after, 
                "tracking_lin_vel": tracking_lin_vel,
                "gravity_vec": gravity_vec,
                "lin_vel_z": lin_vel_z,
                "ang_vel": ang_vel,
                "ang_vel_gyro": ang_vel_gyro,
                **reward_info
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, tracking_lin_vel, gravity_vec, lin_vel_z, ang_vel, ang_vel_gyro, action):
        keep_upright_reward = self._reward_keep_upright(gravity_vec)
        tracking_lin_vel_reward = self._reward_tracking_lin_vel(tracking_lin_vel)
        tracking_ang_vel_reward = self._reward_tracking_ang_vel(ang_vel_gyro)
        upward_orientation_cost = self._cost_upward_orientation(gravity_vec)
        lin_vel_z_cost = self._cost_lin_vel_z(lin_vel_z)
        ang_vel_xy_cost = self._cost_ang_vel_xy(ang_vel)
        ang_vel_gyro_cost = self._cost_ang_vel_gyro(ang_vel_gyro)
        action_cost = self._cost_large_actions(action)

        reward = keep_upright_reward \
               + tracking_lin_vel_reward \
               + tracking_ang_vel_reward \
               + upward_orientation_cost \
               + lin_vel_z_cost \
               + ang_vel_xy_cost \
               + ang_vel_gyro_cost \
               + action_cost

        reward_info = {
            "reward_keep_upright": keep_upright_reward,
            "reward_tracking_lin_vel": tracking_lin_vel_reward,
            "reward_tracking_ang_vel": tracking_ang_vel_reward,
            "cost_upward_orientation": upward_orientation_cost,
            "cost_lin_vel_z": lin_vel_z_cost,
            "cost_ang_vel_xy": ang_vel_xy_cost,
            "cost_ang_vel_gyro": ang_vel_gyro_cost,
            "cost_action": action_cost,
        }
        return reward, reward_info

    def _reward_keep_upright(self, gravity_vec):
        return self._keep_upright_reward_weight * gravity_vec[2]

    def _reward_tracking_lin_vel(self, tracking_lin_vel):
        # movement on the plane
        return self._tracking_lin_vel_reward_weight * np.linalg.norm(tracking_lin_vel)

    def _reward_tracking_ang_vel(self, ang_vel_gyro):
        # yaw
        return self._tracking_ang_vel_reward_weight * np.abs(ang_vel_gyro[2])

    def _cost_upward_orientation(self, gravity_vec):
        return self._upward_orientation_cost_weight * np.linalg.norm(gravity_vec[:2])

    def _cost_lin_vel_z(self, lin_vel_z):
        return self._lin_vel_z_cost_weight * np.abs(lin_vel_z)

    def _cost_ang_vel_xy(self, ang_vel):
        return self._ang_vel_xy_cost_weight * np.linalg.norm(ang_vel[:2])

    def _cost_ang_vel_gyro(self, ang_vel_gyro):
        return self._ang_vel_gyro_cost_weight * np.linalg.norm(ang_vel_gyro[:2])

    def _cost_large_actions(self, action):
        return self._action_cost_weight * np.sum(np.square(action))

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    # Sensor readings. Adapted from mujoco playground.

    def get_gravity(self, data):
        """Return the gravity vector in the world frame."""
        return self.get_sensor_data(self.model, self.data, "upvector")

    def get_global_linvel(self, data):
        """Return the linear velocity of the robot in the world frame."""
        return self.get_sensor_data(
            self.model, self.data, "global_linvel"
        )

    def get_global_angvel(self, data):
        """Return the angular velocity of the robot in the world frame."""
        return self.get_sensor_data(
            self.model, self.data, "global_angvel"
        )

    def get_local_linvel(self, data):
        """Return the linear velocity of the robot in the local frame."""
        return self.get_sensor_data(
            self.model, self.data, "local_linvel"
        )

    def get_accelerometer(self, data):
        """Return the accelerometer readings in the local frame."""
        return self.get_sensor_data(
            self.model, self.data, "accelerometer"
        )

    def get_gyro(self, data):
        """Return the gyroscope readings in the local frame."""
        return self.get_sensor_data(self.model, self.data, "gyro")

    def get_feet_pos(self, data):
        """Return the position of the feet relative to the trunk."""
        return np.vstack([
            self.get_sensor_data(self.model, self.data, sensor_name)
            for sensor_name in ["FL_pos", "FR_pos", "HL_pos", "HR_pos"]
        ])

    def get_sensor_data(self, model, data, sensor_name: str):
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

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
            "xy_position": self.data.qpos[:2],
            "z_position": self.data.qpos[2],
        }
