import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def load_data_and_calculate_metrics(directory, pattern):
    # List all files matching the pattern for each model
    file_paths = glob.glob(os.path.join(directory, pattern))
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        # total_reward = df['reward'].sum()  # Assuming 'reward' column has the rewards for each step
        total_reward = df['reward']
        data.append(total_reward)
    return data

def analyze_models(base_directory):
    metrics = {}
    # Assuming model file patterns like 'ppo_r*_seed*_ent*_lr*__S_*.csv'
    for i in range(1, 9):  # Assuming 8 models
        pattern = f'{i}ppo_r3_seed*__S_*.csv'
        rewards = load_data_and_calculate_metrics(base_directory, pattern)
        if rewards:
            metrics[f'Model {i}'] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards)
            }
    return metrics

def plot_data(metrics):
    plt.figure(figsize=(10, 6))
    for model, stats in metrics.items():
        plt.errorbar(model, stats['mean'], yerr=stats['std'], fmt='o', label=f'{model} Mean and STD')
    plt.xlabel('Models')
    plt.ylabel('Total Reward')
    plt.title('Performance of Models Across Different Seeds')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
# pattern = '*ppo_r3_seed*__S_*.csv'    
base_directory = '/path/to/your/csv/files'  # Update this path to where your files are stored
metrics = analyze_models(base_directory)
plot_data(metrics)