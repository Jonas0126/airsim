import os
import gymnasium as gym
import argparse
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure


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
    p.add_argument("--hidden", type=int, default=256)

    # --- Episode based checkpoint ---
    p.add_argument("--save_every_ep", type=int, default=50)

    # --- Path ---
    p.add_argument("--log_dir", type=str, default="./ppo_logs")
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    print(f'env_id : {args.env_id}')
    input()
    os.makedirs(args.log_dir, exist_ok=True)
    save_dir = os.path.join(args.log_dir, "episode_models")
    os.makedirs(save_dir, exist_ok=True)

    # Logger：不要 CSV，只要 stdout + tensorboard
    logger = configure(args.log_dir, ["stdout", "tensorboard"])

    # Training env（Gymnasium）
    env = DummyVecEnv([make_env(args.env_id)])
    #env = VecMonitor(env)

    # 建立或載入模型
    policy_kwargs = dict(
        net_arch=[args.hidden, args.hidden]
    )

    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
        model.set_logger(logger)
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
            verbose=2,
            tensorboard_log=args.log_dir,
            device='cpu'
        )
        model.set_logger(logger)

    print("[INFO] Start training...")
    print(f"[PARAM] lr={args.lr}, gamma={args.gamma}, n_steps={args.n_steps}, batch={args.batch_size}")

    episode_counter = 0

    # Episode callback
    def callback(locals_, globals_):
        nonlocal episode_counter
        infos = locals_["infos"]

        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                episode_counter += 1

                print(f"[Episode {episode_counter}] reward={r:.2f}, length={l}")

                # Episode-based checkpoint
                if episode_counter % args.save_every_ep == 0:
                    save_path = os.path.join(save_dir, f"ppo_ep{episode_counter}.zip")
                    model.save(save_path)
                    print(f"[SAVE] {save_path}")

        return True

    # Train
    model.learn(
        total_timesteps=args.total_steps,
        callback=callback
    )

    # Save final model
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n[INFO] Training finished. Final saved to: {final_path}\n")


if __name__ == "__main__":
    main()

