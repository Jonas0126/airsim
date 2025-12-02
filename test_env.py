import envs
import gymnasium as gym
import numpy as np
import time

ENV_ID = 'airsim_drone-v0'
TOTAL_STEPS = 100

def simple_run():
    env = None
    try:
        env = gym.make(ENV_ID)
        obs, info = env.reset()
        print(f'obs: {obs}') 
        done = False
        step_count = 0

        while step_count < TOTAL_STEPS:
            action = env.action_space.sample()
            #action = np.array([0,0,0,0.6])            
            obs, reward, terminated, truncated, info = env.step(action)
                       
            current_position = obs[:3]
            
            print(f"Step {step_count+1:02d} | Position: {current_position}")
            
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
    simple_run()
  
