import envs
import gymnasium as gym
import numpy as np
import time
import random
import torch
import os
ENV_ID = 'airsim_drone-v0'
TOTAL_STEPS = 100

def seed_everything(seed: int):
    """
    Fix random seeds for reproducibility.

    Args:
        seed (int): random seed
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic CUDA (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Seed everything with seed = {seed}")


def simple_run():
    env = None
    try:
        env = gym.make(ENV_ID)
        obs, info = env.reset()
        print(f'obs: {obs}') 
        done = False
        step_count = 0

        while step_count < TOTAL_STEPS:
            # action = env.action_space.sample()
            action = np.array([0,0,0,0.2])            
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_position = obs[:3]
            
            print(f"Step {step_count+1:02d} \nPosition: {current_position} \nObs: {obs}")
            print(f'-----'*10)
            done = terminated or truncated
            step_count += 1
            
            if done:
                obs, info = env.reset()
                print(f'position : {obs[:3]}')
            time.sleep(0.05) 
            

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    seed_everything(42)
    simple_run()
  
