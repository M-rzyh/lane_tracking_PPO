from vehicle import Driver
from controller import Camera, Display, GPS, Keyboard, Lidar
import math

# Constants and parameters
TIME_STEP = 50
UNKNOWN = 99999.99
KP = 0.25
KI = 0.006
KD = 2
FILTER_SIZE = 3

class AutonomousVehicleController:
    def __init__(self):
        self.driver = Driver()
        self.keyboard = Keyboard()
        self.keyboard.enable(TIME_STEP)
        
        self.camera = self.driver.getDevice("camera")
        self.camera.enable(TIME_STEP)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()
        
        self.lidar = self.driver.getDevice("Sick LMS 291")
        self.lidar.enable(TIME_STEP)
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_range = self.lidar.getMaxRange()
        self.lidar_fov = self.lidar.getFov()
        
        self.display = self.driver.getDevice("display")
        self.gps = self.driver.getDevice("gps")
        self.gps.enable(TIME_STEP)
        
        self.speed = 0
        self.steering_angle = 0
        self.manual_steering = 0
        self.autodrive = True
        self.pid_need_reset = False
        
        self.old_values = [0] * FILTER_SIZE
        self.first_call = True
        
        self.print_help()

    def print_help(self):
        print("You can drive this car!")
        print("Select the 3D window and then use the cursor keys to:")
        print("[LEFT]/[RIGHT] - steer")
        print("[UP]/[DOWN] - accelerate/slow down")

    def set_autodrive(self, onoff):
        if self.autodrive == onoff:
            return
        self.autodrive = onoff
        if onoff:
            print("Switching to auto-drive...")
        else:
            print("Switching to manual drive... Hit [A] to return to auto-drive.")

    def set_speed(self, kmh):
        if kmh > 250.0:
            kmh = 250.0
        self.speed = kmh
        print(f"Setting speed to {kmh} km/h")
        self.driver.setCruisingSpeed(kmh)

    def set_steering_angle(self, angle):
        if abs(angle - self.steering_angle) > 0.1:
            angle = self.steering_angle + (0.1 if angle > self.steering_angle else -0.1)
        self.steering_angle = max(min(angle, 0.5), -0.5)
        self.driver.setSteeringAngle(self.steering_angle)

    def change_manual_steer_angle(self, inc):
        self.set_autodrive(False)
        new_angle = self.manual_steering + inc
        if -25.0 <= new_angle <= 25.0:
            self.manual_steering = new_angle
            self.set_steering_angle(self.manual_steering * 0.02)

    def check_keyboard(self):
        key = self.keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                self.set_speed(self.speed + 5.0)
            elif key == Keyboard.DOWN:
                self.set_speed(self.speed - 5.0)
            elif key == Keyboard.RIGHT:
                self.change_manual_steer_angle(1)
            elif key == Keyboard.LEFT:
                self.change_manual_steer_angle(-1)
            elif key == ord('A'):
                self.set_autodrive(True)
            key = self.keyboard.getKey()

    def color_diff(self, a, b):
        return sum(abs(a[i] - b[i]) for i in range(3))

    def process_camera_image(self, image):
        num_pixels = self.camera_width * self.camera_height
        REF = [95, 187, 203]  # BGR format for yellow in Webots
        sumx = 0
        pixel_count = 0
        for x in range(num_pixels):
            pixel = image[x * 4:(x + 1) * 4]  # Assuming BGRA format
            if self.color_diff(pixel, REF) < 30:
                sumx += x % self.camera_width
                pixel_count += 1
        if pixel_count == 0:
            return UNKNOWN
        return ((sumx / pixel_count / self.camera_width) - 0.5) * self.camera_fov

    def filter_angle(self, new_value):
        if self.first_call or new_value == UNKNOWN:
            self.first_call = False
            self.old_values = [0] * FILTER_SIZE
        else:
            self.old_values = self.old_values[1:] + [new_value]
        if new_value == UNKNOWN:
            return UNKNOWN
        return sum(self.old_values) / FILTER_SIZE

    def run(self):
        while self.driver.step() != -1:
            self.check_keyboard()
            if self.autodrive and self.camera:
                camera_image = self.camera.getImage()
                yellow_line_angle = self.filter_angle(self.process_camera_image(camera_image))
                if yellow_line_angle != UNKNOWN:
                    self.set_steering_angle(self.apply_pid(yellow_line_angle))
                else:
                    self.driver.setBrakeIntensity(0.4)  # Lost line, start braking

    def apply_pid(self, yellow_line_angle):
        if self.pid_need_reset:
            self.old_value = yellow_line_angle
            self.integral = 0
            self.pid_need_reset = False
        diff = yellow_line_angle - self.old_value
        if diff * self.old_value < 0:  # Sign change, reset integral
            self.integral = 0
        self.integral += yellow_line_angle
        self.old_value = yellow_line_angle
        return KP * yellow_line_angle + KI * self.integral + KD * diff

if __name__ == "__main__":
    controller = AutonomousVehicleController()
    controller.run()