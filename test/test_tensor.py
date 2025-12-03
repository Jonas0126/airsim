import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# ------------------------------
# 設定 log_dir
log_dir = "./ppo_test_logs"
os.makedirs(log_dir, exist_ok=True)

# 配置 logger，stdout + TensorBoard
logger = configure(log_dir, ["stdout", "tensorboard"])

# ------------------------------
# 建立環境
env = gym.make("CartPole-v1")

# ------------------------------
# 建立模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
)

# 設定 logger
model.set_logger(logger)

# ------------------------------
# 訓練
model.learn(total_timesteps=10_000)  # 可以隨時打開 tensorboard 查看

# ------------------------------
# 保存模型
model.save(os.path.join(log_dir, "final_model.zip"))

print(f"Training finished. Logs are in {log_dir}")
