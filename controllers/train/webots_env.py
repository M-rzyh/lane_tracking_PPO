import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import city
import ImageProcessing as ip

class WebotsEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}
    
    # Initializing the action and observation space!
    def __init__(self,driver=None,camera=None, seed=None):
        super(WebotsEnv, self).__init__() 
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
        
        self.steering = 0  # Initial steering angle
        self.speed = 10.0  # Initial speed
        self.actual_speed = 0
        self.actual_steering = 0
        self.current_step = 0
        self.max_steps = 10000  # adjust as needed
        
        self.none_distance_counter = 0
        self.close_distance_counter = 0
        self.low_speed_counter = 0
        
    def read_image_data(self):
        with open("sensor_data.txt", "rb") as file:
            image_data = file.read()
        return image_data
    
    def step(self, action): # you can add coefficients to the actions!!
        
        if action == 0:
            self.steering -= 0.05
        elif action == 1:
            self.steering += 0.05
        elif action == 2:
            self.steering = 0  # Assuming '2' is to reset steering
        # elif action == 3:
        #     self.speed += 5
        # elif action == 4:
        #     self.speed -= 5
        else:
            print("Invalid action")
            
        # Limit steering and speed
        self.steering = np.clip(self.steering, -0.8, 0.8)
        self.speed = np.clip(self.speed, 0, 30)
        
        # Apply action to Webots driver
        self.driver.setSteeringAngle(self.steering)
        # self.driver.setCruisingSpeed(self.speed)
        self.driver.setCruisingSpeed(10)

        
        # Step Webots simulation
        if self.driver.step() == -1:
            return self.reset(), 0, True, {}

        self.current_step += 1
        
        # Measure current state
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
        
        # new_action = np.array([self.steering, self.speed], dtype=np.float32)
        
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
        reward = self.calculate_reward(self.left_distance_m, self.right_distance_m, self.actual_speed)
        if np.isnan(reward):
            print("NaN in reward:", reward)
        
        # Termination logic
        terminated = self.terminate()
        # terminated = False
        turncated = self.turncated()
        info = {}

        print(f"action: {action}, observation: {observation}, reward: {reward}, terminated: {terminated}, turncated: {turncated}")
        # return observation, reward, done, info
        return observation, reward, terminated, turncated, info
    
    
    def calculate_reward(self, left_distance_meters, right_distance_meters, speed):
        
        # Define the distance thresholds
        safe_distance = 2.0
        warning_distance = 1.0
        alpha_speed = 0.0
        alpha_distance = 1.0
        
        # Check the conditions and assign rewards
        if left_distance_meters > safe_distance and right_distance_meters > safe_distance:
            distance_reward = 1  # Safe between lines
        elif left_distance_meters > warning_distance and right_distance_meters > warning_distance:
            distance_reward = -1  # Close to a line but not critically
        else:
            distance_reward = -2  # Too close to one of the lines

        if speed > 15:
            speed_reward = -1
        elif speed > 5:
            speed_reward = 1
        else:
            speed_reward = -1

        return alpha_speed * speed_reward + alpha_distance * distance_reward


    def reset(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        # Reset environment state 
        # Return initial observation
        
        self.steering = 0
        self.speed = 0
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)
        self.actual_speed = 0
        self.actual_steering = 0
        
        self.driver.step()  # Advance Webots simulation to apply reset
        
        self.current_step = 0
        self.none_distance_counter = 0
        self.close_distance_counter = 0
        self.low_speed_counter = 0
        observation = np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)
        
        info = {}
        return observation, info
        
    def terminate(self):
        # Terminate the environment
        done = False
        safe_pixel_sum = 190  # pixels
        minimum_speed = 0.1  # m/s (stopped threshold)

        # Track consecutive None distances
        if self.left_distance_pixel is None or self.right_distance_pixel is None:
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
        if (self.none_distance_counter >= 10 or
            self.close_distance_counter >= 10 or
            self.low_speed_counter >= 100):
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

    def render(self, mode='console'):
        # Implement rendering for debugging
        if mode == 'console':
            print(f'Step: {self.current_step}')
        else:
            super(WebotsEnv, self).render(mode=mode)
            
            
    def pixel_to_meters(self, pixel_value):
        # pixel_value
        meters = pixel_value / 42.51
        return meters
    
