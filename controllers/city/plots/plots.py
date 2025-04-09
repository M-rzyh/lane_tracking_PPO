import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def smooth_data1(data, window_size=50):
    """Apply a moving average filter to smooth the data."""
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

def smooth_data2(data, alpha=0.1):
    """Apply an exponential moving average filter to smooth the data."""
    return data.ewm(alpha=alpha).mean()

def load_data_and_calculate_metrics(directory, model_number):
    data_frames = []
    seeds = [42, 123]
    print(f"Processing model {model_number} with seeds {seeds}")  # Debugging statement
    
    for seed in seeds:
        # 2ppo_r4_seed42_ent0.11_lr1e-4__S_123
        pattern = f'Unstructured.csv'
        # pattern = rf"{model_number}ppo_r4_seed(42|123)_ent(0\.11|0\.05)_lr(1e-4|3e-4)__S_(42|123)\.csv"
    
        file_paths = glob.glob(os.path.join(directory, pattern))
        print(f"Found {len(file_paths)} files for model {model_number} with seed {seed}")  # Debugging statement
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            if 'reward' in df.columns:
                data_frames.append(df['reward'])  # Assuming 'reward' column holds rewards at each step
            else:
                print(f"File {file_path} does not contain 'reward' column")  # Debugging statement

    # Concatenate data for the same model across different seeds
    if data_frames:
        combined_df = pd.concat(data_frames, axis=1)
        mean_rewards = combined_df.mean(axis=1)  # Average across different seeds
        smoothed_rewards = smooth_data1(mean_rewards)  # Apply smoothing
        sec_smoothed_rewards = smooth_data2(smoothed_rewards)  # Apply second smoothing
        return sec_smoothed_rewards
    else:
        print(f"No valid data frames collected for model {model_number}")  # Debugging statement
    return pd.Series()

def analyze_models(base_directory):
    models_data = {}
    for model_number in range(1, 2):  # Assuming 8 models
        smoothed_rewards = load_data_and_calculate_metrics(base_directory, model_number)
        if not smoothed_rewards.empty:
            models_data[f'Model {model_number}'] = smoothed_rewards
        else:
            print(f"No data to plot for Model {model_number}. Check data files and contents.")  # Debugging statement
    return models_data

def plot_data(models_data):
    if not models_data:
        print("No models to plot. Exiting.")  # Debugging statement
        return

    plt.figure(figsize=(12, 8))
    for model, rewards in models_data.items():
        if not rewards.empty:
            steps = np.arange(len(rewards))
            plt.plot(steps, rewards, label=model,color=('b'))
        else:
            print(f"No rewards data available for {model}.")  # Debugging statement
    plt.xlabel('Steps')
    plt.ylabel('Smoothed Average Reward')
    plt.title('Smoothed Average Rewards Over Time for Each Model Across Seeds 42 and 123')
    plt.legend(title="Models")
    plt.grid(True)
    plt.show()

# # Update this path to where your files are stored
base_directory = '/Users/marziehghayour/Library/CloudStorage/GoogleDrive-marzieh.ghayour.na@gmail.com/My Drive/U of A/ECE 720/Project/Simulation/Map2/controllers/city/plots'
models_data = analyze_models(base_directory)
plot_data(models_data)