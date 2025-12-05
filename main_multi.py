import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import gymnasium as gym
import argparse
import envs
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# 引入 SubprocVecEnv 以支援並行環境 (如果 AirSim 環境允許)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from collections import deque
import airsim

# 預設 AirSim 埠號，AirSim 預設是 41451，每個額外環境增加 1
DEFAULT_PORT = 41451 

def make_env(env_id, port):
    """
    建立一個帶有 Monitor 監測的環境，並傳入特定的 port 參數。
    
    Gymnasium reset -> (obs, info)
    step -> (obs, reward, terminated, truncated, info)
    Monitor 自動產生 info["episode"]
    """
    def _init():
        # *** 假設你的 envs.py 中的環境建構函式可以接受 port 參數 ***
        # 例如：env = gym.make(env_id, port=port)
        # 如果你的環境名稱已經註冊到 gym，可能需要像下面這樣傳入 kwargs
        env = gym.make(env_id, **{'port': port}) 
        env = Monitor(env)
        return env
    return _init


def parse_args():
    p = argparse.ArgumentParser()

    # --- 必要 ---
    p.add_argument("--env_id", type=str, required=True)

    # --- 訓練參數 ---
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=128)
    
    # *** 新增：並行環境的數量 ***
    p.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")

    # --- Episode based checkpoint ---
    p.add_argument("--save_every_ep", type=int, default=10000)

    # --- Path ---
    p.add_argument("--log_dir", type=str, default="./ppo_logs")
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


# # FunctionCallback 保持不變，它已經是 VecEnv 通用的
# class FunctionCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.ep_rewards = deque(maxlen=100)
#         self.ep_lengths = deque(maxlen=100)

#     def _on_step(self) -> bool:
#         infos = self.locals["infos"]
#         for info in infos:
#             if "episode" in info:
#                 r = info["episode"]["r"]
#                 l = info["episode"]["l"]
#                 self.ep_rewards.append(r)
#                 self.ep_lengths.append(l)

#                 self.logger.record("rollout/ep_reward", r)
#                 self.logger.record("rollout/ep_length", l)

#         mean_r = sum(self.ep_rewards) / len(self.ep_rewards) if self.ep_rewards else 0
#         mean_l = sum(self.ep_lengths) / len(self.ep_lengths) if self.ep_lengths else 0
#         if self.ep_rewards:
#             self.logger.record("rollout/mean_reward", mean_r)
#             self.logger.record("rollout/mean_length", mean_l)

#         # 每 100 steps dump 到 tensorboard
#         if self.num_timesteps % 100 == 0:
#             self.logger.dump(self.num_timesteps)

#         # 同時輸出到 stdout
#         sys.stdout.write(f'\rtotal steps = {self.num_timesteps} | MeanR={mean_r:.2f} | MeanL={mean_l:.1f} | ')
#         sys.stdout.flush()

#         return True


class CheckpointCallback(BaseCallback):
    """
    每隔 save_every_ep 個 episode 保存一次模型。
    """
    def __init__(self, save_dir, save_every_ep=10000, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_every_ep = save_every_ep
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                if self.episode_count % self.save_every_ep == 0:
                    save_path = os.path.join(
                        self.save_dir,
                        f"checkpoint_ep{self.episode_count}.zip"
                    )
                    self.model.save(save_path)
                    if self.verbose:
                        print(f"[INFO] Checkpoint saved: {save_path}")
        return True


def main():
    args = parse_args()
    print(f'env_id : {args.env_id}')
    print(f'num_envs : {args.num_envs}')
    os.makedirs(args.log_dir, exist_ok=True)
    save_dir = os.path.join(args.log_dir, "episode_models")
    os.makedirs(save_dir, exist_ok=True)
    cb = CheckpointCallback(save_dir=save_dir, save_every_ep=args.save_every_ep)
    input(f'press enter to start!')

    
    # 建立多個環境的建構函式列表
    # 訓練環境 (Training envs)
    env_fns = [make_env(args.env_id, DEFAULT_PORT + i) for i in range(args.num_envs)]
    
    # *** 選擇 VecEnv 類型 ***
    # 推薦使用 SubprocVecEnv 來實現真正的並行 (multi-processing)，
    # 但如果 AirSim 連線跨進程有問題，請改回 DummyVecEnv。
    env = SubprocVecEnv(env_fns) 
    # env = DummyVecEnv(env_fns) # 使用 DummyVecEnv 模擬多環境 (單進程)
    
    
    # # 評估環境 (Evaluation env) - 通常只需要一個
    # eval_env_port = DEFAULT_PORT + args.num_envs # 確保評估環境使用一個不衝突的 port
    # env_eval = DummyVecEnv([make_env(args.env_id, eval_env_port)])


    # eval_callback = EvalCallback(
    #     env_eval,
    #     best_model_save_path='./logs',
    #     n_eval_episodes=5,
    #     eval_freq=10000 // args.num_envs, # 調整頻率以匹配 timesteps
    #     deterministic=True,
    #     render=False,
    #     verbose=0,
    # )


    # 建立或載入模型
    policy_kwargs = dict(
        net_arch=[args.hidden, args.hidden]
    )

    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
        
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            gamma=args.gamma,
            n_steps=args.n_steps,
            gae_lambda=0.95,
            ent_coef=0.02,
            vf_coef=0.3,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=args.log_dir,
            device='cpu'
        )
        

    print("[INFO] Start training...")
    print(f"[PARAM] lr={args.lr}, gamma={args.gamma}, n_steps={args.n_steps}, batch={args.batch_size}, num_envs={args.num_envs}")

    # cb = FunctionCallback()
    # cb_list = CallbackList([cb, eval_callback])
    
    # Train
    model.learn(
        total_timesteps=args.total_steps,
        callback=cb
    )

    # Save final model
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n[INFO] Training finished. Final saved to: {final_path}\n")


if __name__ == "__main__":
    main()