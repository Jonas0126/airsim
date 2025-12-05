from gymnasium.envs.registration import register
from .airsim_drone_env import AirSimDroneEnv
register(
    id="airsim_drone-v0",
    entry_point="envs.airsim_drone_env:AirSimDroneEnv"
)
register(
    id="airsim_vec-v0",
    entry_point="envs.airsim_simple_env:AirSimSimpleTestEnv"
)
register(
    id="airsim_sig-v0",
    entry_point="envs.airsim_single_drone:AirSimSingleDroneEnv"
)
register(
    id="airsim_sig_Genesis-v0",
    entry_point="envs.airsim_sig_drone_from_gengesis:AirSimSingleDroneEnvgGenesis"
)