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

# supervisor = Supervisor()

# robot_node = supervisor.getSelf()
# translation_field = robot_node.getField("translation")
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

env = WebotsEnv(driver, camera=topcamera)


# Training -------------------------------------
observation, info = env.reset()  

# Initialize PPO model
print("Starting PPO Training...")
model = PPO("MlpPolicy", env, ent_coef=0.1, device="cpu", verbose=1)
model.learn(total_timesteps=10000)
print("Training Completed.")

# Save trained model
model.save("ppo_webots_model")
print("Model Saved.")

# Load the saved model
model = PPO.load("ppo_webots_model", device="cpu")  # or "mps" if using Mac GPU    
    
# def save_image_data(image_data):
    # try:
        # with open("sensor_data.txt", "wb") as file:
            # file.write(image_data)
    # except IOError as e:
        # print(f"Failed to save image data: {e}")
        
while driver.step() != -1:
    observation, info = env.reset()  

    # topimage = topcamera.getImage()
    # frontimage = frontcamera.getImage()
    # image = topimage
    # save_image_data(image)
   
    # Get action from trained model
# action, _ = model.predict(observation, deterministic=True)
# observation, info = env.reset()  

    # Take step in the environment
# observation, reward, terminated, truncated, info = env.step(action)

    # Print step details
# print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    # Reset environment if episode ends
# if terminated or truncated:
    # observation, info = env.reset()

            
    
    # current speed and acceleration
    # current_speed = round(driver.getCurrentSpeed(),2)
    # print(f"Current speed: {current_speed} m/s")
    # current_acc = ACC.getValues()
    # print(f"Current acc: {current_acc}")
    
    # GPS
    # gps_position = gps.getValues()
    # print(f"GPS Position: {gps_position}")  # Prints [x, y, z] coordinates
    
    # Steering Setup
    # driver.setSteeringAngle(0)
    # current_steering = round(driver.getSteeringAngle(),2)
    
    # Keyboard Setup
    # key = keyboard.getKey()
            
    # while key != -1:
        # if key == Keyboard.UP:
            # driver.setCruisingSpeed(current_speed + 5)
        # elif key == Keyboard.DOWN:
            # driver.setCruisingSpeed(max(0, current_speed - 5))
        # elif key == Keyboard.LEFT:
            # current_steering = driver.getSteeringAngle()
            # driver.setSteeringAngle(current_steering - 0.1)  # Fine-tune this value as needed
        # elif key == Keyboard.RIGHT:
            # current_steering = driver.getSteeringAngle()
            # driver.setSteeringAngle(current_steering + 0.1)  # Fine-tune this value as needed
        # key = keyboard.getKey()    # Reset brake and recenter steering after adjustments
    
    
    # action, _ = model.predict(observation)
    # observation, reward, done, info = env.step(action)

    # print(f"Action taken: {action}, Reward: {reward}")

    # if done:
        # observation = env.reset()
    
    # observation, reward, done, info = env.step(env.action_space.sample())
    # print(f"Obs: {observation}, Reward: {reward}, Done: {done}")    # print(obs)
    driver.setBrakeIntensity(0)
    pass

# 