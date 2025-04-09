import numpy as np

# This will store the processed LiDAR data in memory
processed_lidar_data = None

def save(data):
    global processed_lidar_data
    processed_lidar_data = data

def get_processed_data():
    return processed_lidar_data