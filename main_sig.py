import os
import gymnasium as gym
import argparse
import envs
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from collections import deque
import airsim
def make_env(env_id):
    """
    Gymnasium reset -> (obs, info)
    step -> (obs, reward, terminated, truncated, info)
    Monitor 自動產生 info["episode"]
    """
    def _init():
        env = gym.make(env_id)
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

    # --- Episode based checkpoint ---
    p.add_argument("--save_every_ep", type=int, default=50)

    # --- Path ---
    p.add_argument("--log_dir", type=str, default="./ppo_logs")
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()



class FunctionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                self.ep_rewards.append(r)
                self.ep_lengths.append(l)

                
                self.logger.record("rollout/ep_reward", r)
                self.logger.record("rollout/ep_length", l)

        mean_r = sum(self.ep_rewards) / len(self.ep_rewards) if self.ep_rewards else 0
        mean_l = sum(self.ep_lengths) / len(self.ep_lengths) if self.ep_lengths else 0
        if self.ep_rewards:
            self.logger.record("rollout/mean_reward", mean_r)
            self.logger.record("rollout/mean_length", mean_l)

        # 每 100 steps dump 到 tensorboard
        if self.num_timesteps % 100 == 0:
            self.logger.dump(self.num_timesteps)

        # 同時輸出到 stdout
        sys.stdout.write(f'\rtotal steps = {self.num_timesteps} | MeanR={mean_r:.2f} | MeanL={mean_l:.1f} | ')
        sys.stdout.flush()

        return True


def main():
    args = parse_args()
    print(f'env_id : {args.env_id}')
    os.makedirs(args.log_dir, exist_ok=True)
    save_dir = os.path.join(args.log_dir, "episode_models")
    os.makedirs(save_dir, exist_ok=True)



    
    input(f'press enter to start!')



    
  



    # # Training env（Gymnasium）
    env = DummyVecEnv([make_env(args.env_id)])


    env_eval = DummyVecEnv([make_env(args.env_id)])


    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path='./logs',
        n_eval_episodes=5,
        eval_freq=1000,
        deterministic=True,
        render=False,
        verbose=0,
    )


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
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=args.log_dir,
            device='cpu'
        )
        

    print("[INFO] Start training...")
    print(f"[PARAM] lr={args.lr}, gamma={args.gamma}, n_steps={args.n_steps}, batch={args.batch_size}")


    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)

    cb = FunctionCallback()
    cb_list = CallbackList([cb, eval_callback])
    # Train
    model.learn(
        total_timesteps=args.total_steps,
        callback=cb_list,
    
    )

    # Save final model
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n[INFO] Training finished. Final saved to: {final_path}\n")


if __name__ == "__main__":
    main()
