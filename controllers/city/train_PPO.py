from stable_baselines3 import PPO
from webots_env import WebotsEnv

def train_model():
    env = WebotsEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("ppo_webots_model")

if __name__ == "__main__":
    train_model()