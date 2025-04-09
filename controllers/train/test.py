from vehicle import Driver
from controller import Camera
from webots_env import WebotsEnv
from stable_baselines3 import PPO

# Set up driver and sensors
driver = Driver()
topcamera = driver.getDevice('topcamera')
topcamera.enable(int(driver.getBasicTimeStep()))

# Create environment and load model
env = WebotsEnv(driver=driver, camera=topcamera)
model = PPO.load("ppo_webots_model", env=env)

# Reset environment
obs, info = env.reset()

# Run policy
while driver.step() != -1:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    
    if terminated or truncated:
        obs, info = env.reset()