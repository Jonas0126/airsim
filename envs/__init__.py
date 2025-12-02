from gymnasium.envs.registration import register
from .airsim_drone_env import AirSimDroneEnv
register(
    id="airsim_drone-v0",
    entry_point="envs.airsim_drone_env:AirSimDroneEnv"
)
