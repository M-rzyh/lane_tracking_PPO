from vehicle import Driver
from controller import Camera, Display, GPS, Keyboard, Lidar
import math
import matplotlib.pyplot as plt
import numpy as np
import ImageProcessing as ip
# import sys
# print(sys.path)
import lidar
from webots_env import WebotsEnv
import gymnasium as gym
from stable_baselines3 import PPO
from controller import Supervisor


driver = Driver()

topcamera = driver.getDevice('topcamera')
topcamera.enable(int(driver.getBasicTimeStep()))

# frontcamera = driver.getDevice('frontcamera')
# frontcamera.enable(int(driver.getBasicTimeStep()))

ACC = driver.getDevice('accelerometer')
ACC.enable(int(driver.getBasicTimeStep()))

gps = driver.getDevice("gps")
gps.enable(int(driver.getBasicTimeStep()))

lidar = driver.getDevice('lidar')
lidar.enable(int(driver.getBasicTimeStep()))


keyboard = Keyboard()
keyboard.enable(int(driver.getBasicTimeStep()))

env = WebotsEnv(driver, camera=topcamera, seed = 42)


# Training -------------------------------------
observation, info = env.reset()  

# print("Starting PPO Training...")
# model = PPO("MlpPolicy", env, ent_coef=0.11,learning_rate = 1e-4, device="cpu", verbose=1)
# model.set_random_seed(42)
# model.learn(total_timesteps=70000)
# print("Training Completed.")

# model.save("")
# print("Model Saved.")

    
         
# while driver.step() != -1:
    # observation, info = env.reset()  
    # driver.setBrakeIntensity(0)
    # pass
    

# Learning ----------------------------------------

# Load the trained model
print("Loading trained model...")
# model = PPO.load("ppo_webots_model", device="cpu")
# model = PPO.load("ppo_webots_model2", device="cpu")
# model = PPO.load("ppo_webots_model_imagechange", device="cpu")
# model = PPO.load("ppo_webots_model_reward1", device="cpu")
# model = PPO.load("ppo_webots_model_reward2", device="cpu")
# model = PPO.load("ppo_webots_model_reward3", device="cpu")
# model = PPO.load("ppo_webots_model_reward3_without_speed", device="cpu")
# model = PPO.load("ppo_webots_model_reward3_without_speed_recordoneline", device="cpu")# smooth turns but not great centering - started near turn for training.
# model = PPO.load("ppo_webots_model_reward4_without_speed_recordoneline", device = "cpu")# isolating - when centering < 0 

# model = PPO.load("ppo_webots_model_reward3_without_speed_recordoneline2", device="cpu")# r3 ent:0.11 lr:3e-4 -  the best of 3



model = PPO.load("ppo_webots_model_reward3_without_speed_recordoneline2", device = "cpu")
print("Model loaded.")

# Evaluation tracking
episode_rewards = []
current_reward = 0
episode = 0

while driver.step() != -1:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    current_reward += reward

    if terminated or truncated:
        # episode_rewards.append(current_reward)
        print(f" Episode {episode + 1} finished with total reward: {current_reward:.2f}")
        current_reward = 0
        episode += 1
        # env.log_episode_metrics()
        observation, info = env.reset()

    driver.setBrakeIntensity(0)

    if episode >= 2:
        break

# Summary
# print("Evaluation complete.")
# print("Episode rewards:", episode_rewards)
# print("Average reward:", np.mean(episode_rewards))
# np.savetxt("evaluation_rewards.txt", episode_rewards)




# import matplotlib.pyplot as plt
# import pandas as pd

# Storage structures
# episode_data = []
# lane_deviation_all = []
# steering_smoothness_all = []

# current_reward = 0
# lane_deviation_episode = []
# steering_changes = []
# prev_steering = None
# episode = 0
# N_eval = 5  # Total episodes to run

# model = PPO.load("ppo_r3_seed42_ent0.11_lr1e-4", device = "cpu")

# while driver.step() != -1:
    # action, _ = model.predict(observation, deterministic=True)
    # observation, reward, terminated, truncated, info = env.step(action)
    # current_reward += reward

    # Steering smoothness (assuming 1D steering)
    # steering = action[0]
    # if prev_steering is not None:
        # steering_changes.append(steering - prev_steering)
    # prev_steering = steering

    # Lane deviation tracking (assuming info contains distances)
    # lane_dev = abs(info["distance_left"] - info["distance_right"])
    # lane_deviation_episode.append(lane_dev)

    # if terminated or truncated:
        # Compute metrics
        # avg_lane_dev = np.mean(lane_deviation_episode)
        # smoothness = np.var(steering_changes)

        # episode_data.append({
            # "episode": episode + 1,
            # "total_reward": current_reward,
            # "avg_lane_deviation": avg_lane_dev,
            # "steering_variance": smoothness
        # })

        # print(f"Episode {episode + 1} - Reward: {current_reward:.2f} | Lane Dev: {avg_lane_dev:.3f} | Smoothness: {smoothness:.4f}")

        # Reset episode
        # current_reward = 0
        # episode += 1
        # lane_deviation_episode = []
        # steering_changes = []
        # prev_steering = None
        # observation, info = env.reset()

        # if episode >= N_eval:
            # break

# Convert to DataFrame
# df_eval = pd.DataFrame(episode_data)
# df_eval.to_csv("ppo_eval_metrics1.csv", index=False)
# print("\nEvaluation metrics saved to 'ppo_eval_metrics1.csv'")







