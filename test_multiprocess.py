# test_multiprocess.py (修正後的版本)

import airsim
import numpy as np
import time
import os
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
import envs

# ----------------- 1. 定義標準環境工廠 -----------------
# 這是 SubprocVecEnv 期望的格式：一個無參數的函數返回一個已初始化環境
def make_env(rank: int, drone_id: str):
    """
    創建一個環境工廠，並在內部初始化 Monitor。
    """
    def _init():
        # 這裡不傳遞 client，環境會在子進程中自行建立
        env = gym.make("airsim_vec-v0", drone_id=drone_id)
        # env = AirSimSimpleTestEnv(drone_id=drone_id) 
        
        # ⚠️ 關鍵：Monitor 必須在 SubprocVecEnv 內部的每個環境中
        env = Monitor(env) 
        return env
    return _init

# ----------------- 2. 測試主函數 -----------------
def main_test():
    DRONE_IDS = ["Drone1", "Drone2"]
    N_ENVS = len(DRONE_IDS)
    
    # ... (全局 AirSim Reset 保持不變) ...
    global_client = airsim.MultirotorClient()
    global_client.confirmConnection()
    global_client.reset() 
    time.sleep(1) 
    
    # 建立環境工廠列表 (每個工廠函數都帶有特定的 drone_id)
    # 這裡我們使用 enumerate 的 index 作為 rank (進程編號)
    env_fns = [make_env(i, drone_id) for i, drone_id in enumerate(DRONE_IDS)]
    
    # 3. 建立 SubprocVecEnv
    # 使用 'spawn' 在 Ubuntu 上更穩定
    env_base = SubprocVecEnv(env_fns, start_method='spawn') 
    
    # ⚠️ 關鍵：在最外層使用 VecMonitor
    # VecMonitor 確保 SubprocVecEnv 的輸出格式和 Monitor 兼容。
    env = VecMonitor(env_base)
    
    print("SubprocVecEnv with VecMonitor 建立成功。")
    
    # 4. 重置所有環境
    # env.reset() 會在所有子進程中執行 AirSimSimpleTestEnv.__init__
    obs, info = env.reset() # ⚠️ VecEnv 的 reset 格式是 (obs, info)

    start_time = time.time()
    
    # 5. 測試並行步驟
    action = np.array([[-2.0] for _ in range(N_ENVS)], dtype=np.float32)
    
    print(f"\n--- 發送並行動作 ({N_ENVS} 動作) ---")
    
    for i in range(5):
        # 執行一步：現在的 VecEnv 輸出是穩定的 5 元組
        # 即使在極端情況下，VecMonitor/SubprocVecEnv 也會確保 info 字典正確傳輸
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: D1 Z={obs[0, 0]:.2f}, D2 Z={obs[1, 0]:.2f}...")
        
        if np.any(terminated):
             print(f"無人機達到目標高度，測試結束。")
             break
    
    end_time = time.time()
    # 6. 驗證結果
    if np.all(obs < -1.0):
        print("\n✅ 成功：所有無人機在並行操作中都向上移動了！")
        print(f"總耗時：{end_time - start_time:.2f} 秒")
    else:
        print("\n❌ 失敗：無人機 Z 軸位置無明顯變化或與預期不符。")

    # 7. 清理
    env.close()
    global_client.reset() 
    print("--- 測試結束 ---")


if __name__ == "__main__":
    # ⚠️ 在 Linux/Ubuntu 中，多進程啟動點必須放在 if __name__ == "__main__": 內
    main_test()