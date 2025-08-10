Changes made in order to add new environments.

1, In the file `envs/__init__py`, added the following lines:

```
register(
    id="Spot-v0",
    entry_point="gymnasium.envs.mujoco.spot_v0:SpotEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
```

2, In the folder `envs/mujoco`, created a file `spot_v0.py`.

3, In the folder `envs/mujoco/assets`, added two files, `spot_v0.xml` and `spot_scene_v0.xml`.

4, In the folder `envs/mujoco/assets`, added a folder `assets` that containts `.obj` files for the spot.