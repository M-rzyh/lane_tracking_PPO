import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import city
import ImageProcessing as ip
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
seed = 42
class WebotsEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}
    
    # Initializing the action and observation space!
    def __init__(self,driver=None,camera=None, seed=seed):
        super(WebotsEnv, self).__init__() 
        
        self.episode_count = 0
        self.total_count = 0
        self.final_episode_count = 0
        
        self.driver = driver
        self.camera = camera
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        
        N = 3 # Number of discrete actions! 0: steer left, 1: steer right, 2: steer straight, speedup, slowdown
        self.action_space = spaces.Discrete(N)
          
        # Example for using image as input:
        self.observation_space = spaces.Box(
            
            low=np.array([0.0, 0.0, 0.0, -0.8],dtype=np.float32),  # Minimum values for left distance, right distance, angle: Add later, speed, steering
            high=np.array([140.0, 140.0, 30.0, 0.8], dtype=np.float32),  # Maximum values for left distance, right distance, angle: Add later, speed, steering
            dtype=np.float32
        )
        
        #steering initialization
        self.steering = 0  # Initial steering angle
        self.prev_steering = 0
        self.actual_steering = 0

        #speed initialization
        self.speed = 0.0  # Initial speed
        self.actual_speed = 0
        self.low_speed_counter = 0

        #step initialization
        self.current_step = 0
        self.max_steps = 2500  # adjust as needed
        
        #distance initialization
        self.none_distance_counter = 0
        self.close_distance_counter = 0
        
        # reward function initializations
        self.steering_record = []
        self.total_rewards = []
        self.centering_track = []
        self.steering_track = []
        
        self.metrics_log = []
        self.metric_log_path = os.path.join("logs_new", "ppo_eval_metrics1.csv")
        os.makedirs("logs_new", exist_ok=True)
        self.reset_episode_data()
        
    def step(self, action): # you can add coefficients to the actions!!
        
        if action == 0:
            self.steering -= 0.05
        elif action == 1:
            self.steering += 0.05
        elif action == 2:
            self.steering = 0  # Assuming '2' is to reset steering
        # elif action == 3:
        #     self.speed += 2
        # elif action == 4:
        #     self.speed -= 2
        else:
            print("Invalid action")
            
        # Limit steering and speed - clipping
        self.steering = np.clip(self.steering, -0.8, 0.8)
        self.speed = np.clip(self.speed, 0, 30)
        
        # Apply action to Webots driver
        self.driver.setSteeringAngle(self.steering)
        # self.driver.setCruisingSpeed(self.speed)
        self.driver.setCruisingSpeed(15)

        
        # Step Webots simulation
        if self.driver.step() == -1:
            return self.reset(), 0, True, {}

                
        # Measure current state of steering and speed
        self.actual_speed = round(self.driver.getCurrentSpeed(), 2)
        self.actual_steering = round(self.driver.getSteeringAngle(), 2)
                
        # Process image data
        image_data = self.camera.getImage()
        ip.lines = ip.process_image(image_data)
        # self.image_data = self.read_image_data()  # Read image data from the file
        # ip.lines = ip.process_image(self.image_data)
        self.left_distance_pixel, self.right_distance_pixel = ip.classify_and_measure_distances_new(ip.lines, 300)

        # Convert pixels to meters for observations
        self.left_distance_m = self.pixel_to_meters(self.left_distance_pixel)
        self.right_distance_m = self.pixel_to_meters(self.right_distance_pixel)
        
        
        # Construct observation
        observation = np.array([
            self.left_distance_m, 
            self.right_distance_m, 
            self.actual_speed if not np.isnan(self.actual_speed) else 0.0, 
            self.actual_steering if not np.isnan(self.actual_steering) else 0.0
        ], dtype=np.float32)
        if np.isnan(observation).any():
            print("NaN in observation:", observation)
            
        # Calculate reward
        # reward = self.calculate_reward(self.left_distance_m, self.right_distance_m, self.actual_speed)
        # reward = self.calculate_reward2(self.left_distance_m, self.right_distance_m, self.actual_speed)
        # reward = self.calculate_reward3(self.left_distance_m, self.right_distance_m, self.actual_speed)
        reward = self.calculate_reward4(self.left_distance_m, self.right_distance_m, self.actual_speed)
        
        if np.isnan(reward):
            print("NaN in reward:", reward)
        
        # Termination logic
        terminated = self.terminate()
        # terminated = False
        turncated = self.turncated()
        info = {}

        self.current_step += 1
        self.total_count += 1
        print(f"Episode: {self.episode_count}, Step: {self.current_step}, Total Count: {self.total_count}")
        print(f"action: {action}, observation: {observation}, reward: {reward}, terminated: {terminated}, turncated: {turncated}")
        self.final_episode_count = self.episode_count
        
        # Store rewards and distances for analysis
        self.total_rewards.append(reward)
        self.centering_track.append(self.left_distance_m)
        self.steering_track.append(self.right_distance_m)
        
        self.lane_deviation = abs(self.left_distance_m - self.right_distance_m)
        self.log_step_metrics(reward, self.lane_deviation, self.steering, self.current_step, self.episode_count)
        
        return observation, reward, terminated, turncated, info 
    
    def calculate_reward(self, left_distance_meters, right_distance_meters, speed):
        
        # Define the distance thresholds
        safe_distance = 1.5
        warning_distance = 1.0
        
        # coefficients for the reward function
        alpha_speed = 0.0
        alpha_distance = 1.0
        
        # Check the conditions and assign rewards
        if left_distance_meters > safe_distance and right_distance_meters > safe_distance:
            distance_reward = 1  # Safe between lines
        elif left_distance_meters > warning_distance and right_distance_meters > warning_distance:
            distance_reward = 0  # Close to a line but not critically
        else:
            distance_reward = -1  # Too close to one of the lines

        if speed > 15:
            speed_reward = -1
        elif speed > 5:
            speed_reward = 1
        else:
            speed_reward = -1

        return alpha_speed * speed_reward + alpha_distance * distance_reward

    def calculate_reward2(self, left_distance_meters, right_distance_meters, speed):
        # Total width of the lane
        lane_width = 5.0  # in meters

        # Deviation from the center (should be 0 if perfectly centered)
        deviation = abs(left_distance_meters - right_distance_meters)

        # Normalize deviation: 0 (perfect center), 1 (hugging a line)
        normalized_deviation = deviation / lane_width

        # Reward for staying centered: max 1.0, decreases as deviation increases
        center_reward = 1.0 - normalized_deviation  # in range [0,1]

        # Penalize being too close to either side
        margin = 0.8  # in meters
        if left_distance_meters < margin or right_distance_meters < margin:
            edge_penalty = -1.0
        else:
            edge_penalty = 0.0

        # Reward for moving at an acceptable speed (encourage smooth motion)
        if speed < 1.0:
            speed_reward = -0.5
        elif speed > 15.0:
            speed_reward = -0.5
        else:
            speed_reward = 0.5  # bonus for staying in an optimal range

        total_reward = 2.0 * center_reward + edge_penalty + 0.5 * speed_reward
        return total_reward

    def calculate_reward3(self, left_distance_meters, right_distance_meters, speed):
        
        # Centering - cosine reward
        if left_distance_meters is None or right_distance_meters is None:
            lane_width = 5.0
        else:
            lane_width  = left_distance_meters + right_distance_meters

        alpha = 1
        centering_norm = ((lane_width - abs(left_distance_meters - right_distance_meters))*math.pi/(lane_width))
        if abs(centering_norm) == math.nan or abs(centering_norm) == math.inf:
            centering_norm = 0
        
        print(f"Centering Norm: {centering_norm}")
        self.centering_reward = -1 * math.cos(centering_norm)
        print(f"Centering Reward: {self.centering_reward}")
        if self.close_distance_counter >= 20:
            self.centering_reward = -3   
            
        #Steering - variance reward
        beta = 1
        self.steering_record.append(self.steering)
        variance = np.var(self.steering_record[-10:])
        self.steering_reward = -50* variance
        print(f"Steering Reward: {self.steering_reward}")
        
        # Reward calculation
        self.reward = (alpha * self.centering_reward + beta * self.steering_reward)

        return self.reward
    
    def calculate_reward4(self, left_distance_meters, right_distance_meters, speed):
        
        print("Reward 4")
        # Centering - cosine reward
        if left_distance_meters is None or right_distance_meters is None:
            lane_width = 5.0
        else:
            lane_width  = left_distance_meters + right_distance_meters

        alpha = 1
        centering_norm = ((lane_width - abs(left_distance_meters - right_distance_meters))*math.pi/(lane_width))
        if abs(centering_norm) == math.nan or abs(centering_norm) == math.inf:
            centering_norm = 0
        
        print(f"Centering Norm: {centering_norm}")
        self.centering_reward = -1 * math.cos(centering_norm)
        print(f"Centering Reward: {self.centering_reward}")
        if self.close_distance_counter >= 20:
            self.centering_reward = -3   
            
        # Steering reward - mean reward
        beta = 1
        self.steering_reward = 0
        if self.centering_reward > 0:
            self.steering_reward = 0
            self.steering_record = []
        else:
            self.steering_record.append(self.steering)
            self.steering_reward = np.mean(self.steering_record)
            self.steering_record = self.steering_record[-10:]  # Keep the last 10 values
        print(f"Steering Reward: {self.steering_reward}")
        
        self.reward = (alpha * self.centering_reward + beta * self.steering_reward)
        # self.total_rewards.append(self.reward)
        # self.centering_track.append(self.centering_reward)
        # self.steering_track.append(self.steering)
        return self.reward
        
    def reset(self, seed=seed):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        self.episode_count += 1
        
        self.log_episode_metrics(self.episode_count)
        
        self.steering = 0
        self.speed = 0
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)
        self.actual_speed = 0
        self.actual_steering = 0
        
        self.steering_record = []
        
        with open("reset_trigger.txt", "w") as f:
            f.write("reset")

        self.current_step = 0
        self.none_distance_counter = 0
        self.close_distance_counter = 0
        self.low_speed_counter = 0
        observation = np.array([2.5, 2.5, 0.0, 0.0], dtype=np.float32)
        # print("Final_Episode_Count: ", self.final_episode_count)

        # Clear for next episode
        self.total_rewards.clear()
        self.centering_track.clear()
        self.steering_track.clear()
        info = {}
        

        # Reset per-step data for next episode
        self.reset_episode_data()
        
        return observation, info
        
    def terminate(self):
        # Terminate the environment
        done = False
        safe_pixel_sum = 150  # pixels
        minimum_speed = 0.1  # m/s (stopped threshold)

        # Track consecutive None distances
        if self.left_distance_pixel >= 250 or self.right_distance_pixel >= 250:
            self.none_distance_counter += 1
        else:
            self.none_distance_counter = 0

        # Track sum of distances being too small (too close to lane lines)
        if (self.left_distance_pixel is not None and self.right_distance_pixel is not None and 
            (self.left_distance_pixel + self.right_distance_pixel) < safe_pixel_sum):
            self.close_distance_counter += 1
        else:
            self.close_distance_counter = 0

        # Track low speed
        if self.actual_speed < minimum_speed:
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0

        # Check termination conditions over 10 consecutive steps
        if (self.none_distance_counter >= 1000 or
            self.close_distance_counter >= 50 or
            self.low_speed_counter >= 50):
            print("Termination condition met.")
            print(self.low_speed_counter)
            print(self.close_distance_counter)
            print(self.none_distance_counter)
            done = True
        else:
            done = False
            
        return done
     
    def turncated(self):
        done = False
        if self.current_step >= self.max_steps:
            print("Maximum steps reached.")
            done = True
        return done          
            
    def pixel_to_meters(self, pixel_value):
        # pixel_value
        meters = pixel_value / 42.51
        return meters
    
    def log_step_metrics(self, reward, lane_deviation, steering, step_number, episode_count):
        self.step_rewards.append(reward)
        self.step_lane_deviations.append(lane_deviation)
        self.step_steering_variances.append(steering)
        # Store each step's data in the main log
        self.metrics_log.append({
            "episode": episode_count,
            "step": step_number,
            "reward": reward,
            "lane_deviation": lane_deviation,
            "steering_variance": steering
        })
    
    def reset_episode_data(self):
        self.step_rewards = []
        self.step_lane_deviations = []
        self.step_steering_variances = []
        
    def log_episode_metrics(self, final_episode_count):
        # Calculate per-episode statistics
        avg_reward = np.mean(self.step_rewards) if self.step_rewards else 0
        total_reward = sum(self.step_rewards)
        avg_lane_deviation = np.mean(self.step_lane_deviations) if self.step_lane_deviations else 0
        steering_variance = np.var(self.step_steering_variances) if self.step_steering_variances else 0

        # Add episode summary to log
        self.metrics_log.append({
            "episode": final_episode_count,
            "step": "summary",
            "reward": avg_reward,
            "total_reward": total_reward,
            "lane_deviation": avg_lane_deviation,
            "steering_variance": steering_variance
        })
        # Save to CSV
        df = pd.DataFrame(self.metrics_log)
        df.to_csv(self.metric_log_path, index=False)

        print(f"Logged Episode {final_episode_count} - Total Reward: {total_reward:.2f}, "
              f"Avg Reward: {avg_reward:.2f}, Avg Lane Dev: {avg_lane_deviation:.3f}, "
              f"Steering Variance: {steering_variance:.4f}")
        
        
        
    # def log_episode_metrics(self):
    #     if not hasattr(self, "episode_metrics_log"):
    #         self.episode_metrics_log = []
    #         self.metric_log_path = os.path.join("logs", "ppo_eval_metrics.csv")
    #         os.makedirs("logs", exist_ok=True)

    #     avg_lane_deviation = np.mean([abs(l - r) for l, r in zip(self.centering_track, self.steering_track)]) \
    #                          if self.centering_track and self.steering_track else 0
    #     steering_variance = np.var(self.steering_track[-10:]) if self.steering_track else 0
    #     total_reward = sum(self.total_rewards)

    #     self.episode_metrics_log.append({
    #         "episode": self.final_episode_count,
    #         "total_reward": total_reward,
    #         "avg_lane_deviation": avg_lane_deviation,
    #         "steering_variance": steering_variance
    #     })

    #     # Save to CSV every episode
    #     df = pd.DataFrame(self.episode_metrics_log)
    #     df.to_csv(self.metric_log_path, index=False)

    #     print(f"Logged Episode {self.final_episode_count} - Reward: {total_reward:.2f}, "
    #           f"Lane Dev: {avg_lane_deviation:.3f}, Smoothness: {steering_variance:.4f}")
    