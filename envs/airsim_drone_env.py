import airsim
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
class AirSimDroneEnv(gym.Env):

    metadata = {
        'render_modes': [], 
        'render_fps': 30  # 雖然沒有渲染，但最好保留這個鍵
    }

    def __init__(self, max_episode_steps=1000, step_length=0.3):
        super().__init__()
        MAX_ROLL_RATE = 3.0
        MAX_YAW_RATE = 3.0
        MAX_PITCH_RATE = 3.0
        MAX_THRUST = 1.0
        MIN_THRUST = 0
        MAX_COORD = 100
        self.max_episode_steps = max_episode_steps
        self.step_length = step_length
        self.steps = 0

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        
        self.drone = airsim.MultirotorClient("127.0.0.1")
        self.action_space = spaces.Box(low=np.array([-MAX_ROLL_RATE, -MAX_PITCH_RATE, -MAX_YAW_RATE, MIN_THRUST]),
                                       high=np.array([MAX_ROLL_RATE, MAX_PITCH_RATE, MAX_YAW_RATE, MAX_THRUST]))

        low_obs = np.array([-np.inf] * 9)
        
        high_obs = np.array([np.inf] * 9)
        
        self.observation_space = spaces.Box(low=low_obs,
                                            high=high_obs)
        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

    def _get_obs(self):

    
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        
        
        current_position_vec3 = self.drone_state.kinematics_estimated.position
        self.state["position"] = np.array([current_position_vec3.x_val, current_position_vec3.y_val, current_position_vec3.z_val])
        
        
        current_velocity_vec3 = self.drone_state.kinematics_estimated.linear_velocity
        self.state["velocity"] = np.array([current_velocity_vec3.x_val, current_velocity_vec3.y_val, current_velocity_vec3.z_val])

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        
        obs = np.concatenate([
            self.state["position"],    
            self.state["velocity"],    
            self.state["prev_position"] 
        ])
        
        return np.array(obs, dtype=np.float32)      

    def _do_action(self, action):
        r, p, y, t = [float(a) for a in action]
        self.drone.moveByAngleRatesThrottleAsync(
            r,
            p,
            y,
            t,
            self.step_length
        ).join()

    def _compute_reward(self):
        reward = 0
        done = False
        target = np.array([0.0, 0.0, -2.0])

        position = self.state["position"]
        dist_to_target = np.linalg.norm(position - target)
        print(f'poisition : {position}')
        print(f'dist_to_target : {dist_to_target}')
        reward = -dist_to_target + 1.0
        print(f'reward : {reward}')
        print(f'steps : {self.steps}')

        if self.state["collision"]:
            reward = -100
            done = True 

        if self.steps >= self.max_episode_steps:
            done = True

        return reward, done

    def step(self, action):
        self._do_action(action)
        self.steps += 1
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, False, self.state

    def reset(self, seed=None, options=None):
        
        self.steps = 0
        self._setup_flight()
        info = {}

        return self._get_obs(), info

    def close(self):
        pass # 讓方法返回 None    
