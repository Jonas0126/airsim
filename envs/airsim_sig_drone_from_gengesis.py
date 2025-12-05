import airsim
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

class AirSimSingleDroneEnvgGenesis(gym.Env):
    metadata = {
        'render_modes': [], 
        'render_fps': 20
    }

    def __init__(self, max_episode_steps=1000, step_length=0.05, port=41451):
        super().__init__()
        
        # === 1. 設定與 Genesis 邏輯對齊 (Configs matching Genesis logic) ===
        self.step_length = step_length  
        self.max_episode_steps = max_episode_steps
        
        # 觀測值的縮放比例
        self.obs_scales = {
            "rel_pos": 1.0 / 3.0,  
            "lin_vel": 1.0 / 2.0,  
            "ang_vel": 1.0 / 0.25, 
        }
        
        # 獎勵函數的權重
        self.reward_scales = {
            "target": 2.0,   
            "smooth": -0.1,  
            "yaw": 0.5,      
            "angular": -0.05,
            "crash": -10.0   
        }
        
        # 應用 dt 縮放
        for k in self.reward_scales:
            self.reward_scales[k] *= self.step_length

        # === 隨機化參數 (Randomization Parameters) ===
        # 1. 隨機出生點範圍 (NED 座標系)
        # 這裡設定 X, Y 在 +/- 1.0m，Z 必須是負值 (代表高度，例如 -1.5m 到 -2.5m)
        self.init_pos_range = {
            "x": [-1.0, 1.0], 
            "y": [-1.0, 1.0], 
            "z": [-2.5, -1.5] # AirSim NED: 負值是向上
        }
        
        # 2. 隨機目標點範圍 (NED 座標系)
        # 目標點在 X, Y +/- 5.0m，Z 在 -3.0m 到 -5.0m 之間
        self.target_pos_range = {
            "x": [-5.0, 5.0], 
            "y": [-5.0, 5.0], 
            "z": [-5.0, -3.0] 
        }

        # 目標與環境參數
        # 初始目標點將在 reset 時隨機產生
        self.target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
        self.clip_actions = 1.0 
        self.at_target_threshold = 0.5 # 稍微放寬到達目標的閾值

        # === 2. AirSim 連線設定 ===
        self.client = None
        self.port=port
        print(f'AirSim connected!')

        # === 3. 動作空間 ===
        self.num_actions = 4
        self.action_space = spaces.Box(
            low=-self.clip_actions, 
            high=self.clip_actions, 
            shape=(self.num_actions,), 
            dtype=np.float32
        )

        # === 4. 觀測空間 ===
        self.num_obs = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        # 初始化緩衝區 (Buffers)
        self.state = {}
        self.last_actions = np.zeros(self.num_actions, dtype=np.float32)
        self.last_base_pos = np.zeros(3, dtype=np.float32)
        self.last_rel_pos = np.zeros(3, dtype=np.float32)
        
        # self._setup_flight()

    def _init_client(self):
        if self.client is None:
            self.client = airsim.MultirotorClient(port=self.port)
            self.client.confirmConnection()



    # --- 新增目標點隨機採樣函式 ---
    def _resample_target(self):
        """ 隨機採樣新的目標點 """
        x = np.random.uniform(*self.target_pos_range["x"])
        y = np.random.uniform(*self.target_pos_range["y"])
        z = np.random.uniform(*self.target_pos_range["z"])
        self.target_pos = np.array([x, y, z], dtype=np.float32)
        # print(f"New Target: {self.target_pos}")

    

    def _setup_flight(self):
        """ 重置無人機狀態並解鎖 """
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)


    def _get_obs(self):
        """ 獲取環境觀測值，並處理座標轉換 (World Frame -> Body Frame) """
        
        self.drone_state = self.client.getMultirotorState()
        kinematics = self.drone_state.kinematics_estimated

        # 1. 位置 (World Frame)
        pos = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val], dtype=np.float32)
        self.state["base_pos"] = pos
        
        # 2. 四元數 -> 旋轉矩陣
        q_raw = kinematics.orientation
        base_quat = np.array([q_raw.x_val, q_raw.y_val, q_raw.z_val, q_raw.w_val], dtype=np.float32)
        r = R.from_quat(base_quat)
        rot_mat = r.as_matrix()  # 3x3
        self.state["base_rot_mat"] = rot_mat

        # 3. 速度 (轉 Body Frame)
        lin_vel_world = np.array([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val], dtype=np.float32)
        ang_vel_world = np.array([kinematics.angular_velocity.x_val, kinematics.angular_velocity.y_val, kinematics.angular_velocity.z_val], dtype=np.float32)
        
        inv_r = r.inv()
        base_lin_vel = inv_r.apply(lin_vel_world).astype(np.float32)
        base_ang_vel = inv_r.apply(ang_vel_world).astype(np.float32)
        
        self.state["base_lin_vel"] = base_lin_vel
        self.state["base_ang_vel"] = base_ang_vel
        self.state["base_euler"] = r.as_euler('xyz', degrees=True) 

        # 4. 相對位置向量
        self.rel_pos = self.target_pos - pos
        
        # 5. 組合觀測向量 (Concatenate)
        obs = np.concatenate([
            np.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
            rot_mat.flatten(),  # 旋轉矩陣展平為9維
            np.clip(base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
            np.clip(base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
            self.last_actions
        ])
        
        self.state["collision"] = self.client.simGetCollisionInfo().has_collided
        
        return obs.astype(np.float32)

    def _do_action(self, action):
        """ 執行動作：將神經網路輸出映射到馬達 PWM """
        
        action = np.clip(action, -self.clip_actions, self.clip_actions)
        
        base_pwm = 0.6
        scale_pwm = 0.4 
        
        pwm_values = base_pwm + (action * scale_pwm)
        pwm_values = np.clip(pwm_values, 0.0, 1.0) 

        self.client.moveByMotorPWMsAsync(
            float(pwm_values[0]), 
            float(pwm_values[1]), 
            float(pwm_values[2]), 
            float(pwm_values[3]), 
            self.step_length 
        ).join()

    # ------------ Reward Functions 維持不變 ----------------
    def _reward_target(self):
        dist_sq_last = np.sum(np.square(self.last_rel_pos))
        dist_sq_curr = np.sum(np.square(self.rel_pos))
        return dist_sq_last - dist_sq_curr

    def _reward_smooth(self):
        return np.sum(np.square(self.curr_action - self.last_actions))

    def _reward_yaw(self):
        yaw_deg = self.state["base_euler"][2]
        if yaw_deg > 180: yaw_deg -= 360
        yaw_rad = yaw_deg * (math.pi / 180.0)
        yaw_lambda = -2.0 
        return np.exp(yaw_lambda * np.abs(yaw_rad))

    def _reward_angular(self):
        return np.linalg.norm(self.state["base_ang_vel"] / math.pi)

    # def _compute_reward_and_done(self):
    #     """ 整合所有獎勵與終止條件 """
        
    #     terminated = False
        
    #     euler = self.state["base_euler"] 
    #     pos = self.state["base_pos"]
    #     rel_pos = self.rel_pos

    #     # 終止條件 (Crash Condition)
    #     crash_cond = (
    #         (abs(euler[1]) > 80) or  # Pitch 過大
    #         (abs(euler[0]) > 80) or  # Roll 過大
    #         (abs(rel_pos[0]) > 5.0) or # 飛太遠 (X軸)
    #         (abs(rel_pos[1]) > 5.0) or # 飛太遠 (Y軸)
    #         (pos[2] > 0.0) or        # 撞地
    #         self.state["collision"]
    #     )
        
    #     # 檢查是否到達目標 (At Target) - 可以選擇是否在到達目標時終止並給予獎勵
    #     at_target = np.linalg.norm(rel_pos) < self.at_target_threshold
        
    #     # 計算各項 Reward
    #     rew_target = self._reward_target() * self.reward_scales["target"]
    #     rew_smooth = self._reward_smooth() * self.reward_scales["smooth"]
    #     rew_yaw = self._reward_yaw() * self.reward_scales["yaw"]
    #     rew_ang = self._reward_angular() * self.reward_scales["angular"]
        
    #     total_reward = rew_target + rew_smooth + rew_yaw + rew_ang
        
    #     if crash_cond:
    #         terminated = True
    #         total_reward += self.reward_scales["crash"]
        
    #     # 如果到達目標，給予額外獎勵並結束回合 (如果不想終止，可以移除 terminated=True)
    #     if at_target:
    #          total_reward += 5.0 # 到達目標獎勵


    #     if self.steps >= self.max_episode_steps:
    #         terminated = True 

    #     return total_reward, terminated

    def _compute_reward_and_done(self):
        terminated = False
        pos = self.state["base_pos"]
        rel_pos = self.rel_pos
        lin_vel = self.state["base_lin_vel"]

        # --- Crash 判斷 ---
        crash_cond = (
            (abs(self.state["base_euler"][1]) > 80) or
            (abs(self.state["base_euler"][0]) > 80) or
            (abs(rel_pos[0]) > 5.0) or
            (abs(rel_pos[1]) > 5.0) or
            (pos[2] > 0.0) or
            self.state["collision"]
        )
        if crash_cond:
            terminated = True

        dist = np.linalg.norm(rel_pos)
        hover_threshold = 0.5  # 判斷懸停區域

        if dist >= hover_threshold:
            # --- 移動階段：速度越大 reward 越高 ---
            # 計算速度沿目標方向的投影
            direction = rel_pos / (dist + 1e-8)
            speed_toward_target = np.dot(-lin_vel, direction)  # 負號因為 rel_pos = target - pos
            rew_move = self.reward_scales["target"] * np.tanh(speed_toward_target)
            rew_hover = 0.0
        else:
            # --- 懸停階段：速度越小 reward 越高 ---
            rew_hover = self.reward_scales["target"] * (1.0 - np.tanh(np.linalg.norm(lin_vel)))
            rew_move = self.reward_scales["target"] * (1.0 - np.tanh(dist))  # 保持距離接近目標也有 reward

        # --- 平滑性獎勵 ---
        rew_smooth = self._reward_smooth() * self.reward_scales["smooth"]
        # --- Yaw 獎勵 ---
        rew_yaw = self._reward_yaw() * self.reward_scales["yaw"]
        # --- Angular velocity 懲罰 ---
        rew_ang = self._reward_angular() * self.reward_scales["angular"]

        total_reward = rew_move + rew_hover + rew_smooth + rew_yaw + rew_ang

        if crash_cond:
            total_reward += self.reward_scales["crash"]

        if self.steps >= self.max_episode_steps:
            terminated = True

        return total_reward, terminated

    def step(self, action):
        self.curr_action = action
        
        self.last_base_pos[:] = self.state["base_pos"][:]
        self.last_rel_pos[:] = self.rel_pos[:] 

        self._do_action(action)
        self.steps += 1
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward_and_done()
        
        self.last_actions[:] = self.curr_action[:]
        
        info = {
            "dist_to_target": np.linalg.norm(self.rel_pos)
        }

        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._init_client()

        # 1. 重置 AirSim 狀態並解鎖
        self._setup_flight()
        
        # 2. 隨機採樣出生點
        init_x = np.random.uniform(*self.init_pos_range["x"])
        init_y = np.random.uniform(*self.init_pos_range["y"])
        init_z = np.random.uniform(*self.init_pos_range["z"])
        init_pos = airsim.Vector3r(init_x, init_y, init_z)
        
        # 3. 將無人機設置到隨機出生點 ( AirSim 的 set_pose )
        # 初始角度設為 0 (不旋轉)
        init_quat = airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)
        self.client.simSetVehiclePose(airsim.Pose(init_pos, init_quat), ignore_collision=True)
        
        # 4. 隨機採樣目標點
        self._resample_target()
        
        # 5. 重置計數器與緩衝區
        self.steps = 0
        self.last_actions[:] = 0.0
        
        # 6. 獲取初始觀測值
        obs = self._get_obs()
        
        # 7. 重置 "Last" 狀態
        self.last_base_pos[:] = self.state["base_pos"][:]
        self.last_rel_pos[:] = self.rel_pos[:]
        
        return obs, {}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)