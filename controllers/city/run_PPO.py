from stable_baselines3 import PPO
from webots_env import WebotsEnv

def run_model():
    env = WebotsEnv()
    model = PPO.load("ppo_webots_model", env=env)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    run_model()