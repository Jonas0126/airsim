import airsim
import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
class AirSimSingleDroneEnv(gym.Env):

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

        self.lambda1 = 1.0
        self.lambda2 = -2e-4
        self.lambda3 = -1e-4
        self.crash_penalty = 5

        # 修成用參數傳
        self.target_point = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        
        self.curr_action = None
        self.prev_action = None
        self.prev_dist = None
        self.drone_state = None

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print(f'connect confirm!')
        self.max_episode_steps = max_episode_steps
        self.step_length = step_length
        self.steps = 0
        self.episode_steps_counter = 0
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        
        # self.drone = airsim.MultirotorClient()
        # self.drone.confirmConnection() 
        self.action_space = spaces.Box(low=np.array([-MAX_ROLL_RATE, -MAX_PITCH_RATE, -MAX_YAW_RATE, MIN_THRUST]),
                                       high=np.array([MAX_ROLL_RATE, MAX_PITCH_RATE, MAX_YAW_RATE, MAX_THRUST]),
                                       dtype=np.float32)


        
        # observation space = 19D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self._setup_flight()


    def _setup_flight(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        


    def _get_obs(self):
        self.drone_state = self.client.getMultirotorState()

        # position
        pos = self.drone_state.kinematics_estimated.position
        p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        self.state["position"] = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # velocity
        vel = self.drone_state.kinematics_estimated.linear_velocity
        v = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        # quaternion → rotation matrix
        q = self.drone_state.kinematics_estimated.orientation
        quat = np.array([q.x_val, q.y_val, q.z_val, q.w_val])
        rot_mat = R.from_quat(quat).as_matrix().flatten().astype(np.float32)
        R_drone = R.from_quat(quat).as_matrix()

        # 相對目標（body frame）
        p_rel_body = R_drone.T @ (self.target_point - p)
        # normalize
        DIST_SCALE = 20.0
        MAX_SPEED = 10.0
        p_norm = p_rel_body / DIST_SCALE
        v_norm = v / MAX_SPEED

        # concatenate
        obs = np.concatenate([p_norm, v_norm, rot_mat, self.prev_action])


        # collision state
        collision = self.client.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return np.array(obs, dtype=np.float32)
       

    def _do_action(self, action):

        r, p, y, t = [float(a) for a in action]
        print(f'r, p, y, t : {r}, {p}, {y}, {t}')
        self.client.moveByAngleRatesThrottleAsync(
            r,
            p,
            y,
            t,
            self.step_length,
        ).join()
        # time.sleep(1)
        

    def _compute_reward(self):

        done = False
        success = False
        reward = 0
        # === 1. Progress Reward ===

        prev_dist = self.prev_dist
        curr_dist = np.linalg.norm(self.state["position"] - self.target_point)
        # curr_dist = abs(self.state["position"][2] - self.target_point[2])
        r_prog = self.lambda1 * (prev_dist - curr_dist)
        self.prev_dist = curr_dist  # 更新

        # === 2. Command Smoothness Reward ===
        a = self.curr_action
        pa = self.prev_action

        r_cmd = (
            self.lambda2 * np.sum(np.abs(a)) +
            self.lambda3 * np.sum((a - pa) ** 2)
        )

        # === 3. Collision Penalty ===
        collision = self.state["collision"]

        if collision:
            r_crash = self.crash_penalty
        else:
            r_crash = 0

        if self.steps >= 200:
            reward += 20
            success = True
        # Final reward
        reward = r_prog + r_cmd - r_crash

        done = success or collision

        return reward, collision

    def step(self, action):
        self.curr_action = action
        self._do_action(action)
        self.steps += 1
        self.episode_steps_counter += 1
        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.prev_action = self.curr_action
        return obs, reward, done, False, self.state

    def reset(self, seed=None, options=None):
        
        self._setup_flight()
        
        self.drone_state = self.client.getMultirotorState()
        pos = self.drone_state.kinematics_estimated.position
        self.state["position"] = np.array([pos.x_val, pos.y_val, pos.z_val])
        self.prev_dist = np.linalg.norm(self.state["position"] - self.target_point) 
        # self.prev_dist = abs(self.state["position"][2] - self.target_point[2])
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.steps = 0
        info = {}

        return self._get_obs(), info

    def close(self):
        pass # 讓方法返回 None    
