from vehicle import Driver
from controller import Camera
from webots_env import WebotsEnv
from stable_baselines3 import PPO

# Initialize Webots devices manually
driver = Driver()
topcamera = driver.getDevice('topcamera')
topcamera.enable(int(driver.getBasicTimeStep()))

# Create the environment
env = WebotsEnv(driver=driver, camera=topcamera)

# Train the model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # or 'mps' for Mac
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_webots_model")
print("✅ Model trained and saved.")

# Keep Webots simulation running (optional)
while driver.step() != -1:
    pass