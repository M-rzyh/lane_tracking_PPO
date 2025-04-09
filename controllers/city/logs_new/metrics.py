import pandas as pd
import numpy as np
import os
import glob

def calculate_metrics(file_path):
    df = pd.read_csv(file_path)
    print(f"Processing file: {file_path}")
    # Assuming 'total_reward' is a column recording the reward of each episode
    final_performance = df['reward'].mean()

    # Assuming 'dleft' and 'dright' are columns for left and right lane distances for each timestep
    avg_lane_deviation = df['lane_deviation'].mean()


    # # You might have a column that indicates whether the environment is structured or not
    # # Assuming 'environment_type' column and 'reward' column exist
    # if 'environment_type' in df.columns and 'reward' in df.columns:
    #     grouped = df.groupby('environment_type')['reward'].mean()
    #     delta_R = grouped.get('structured', 0) - grouped.get('unstructured', 0)
    # else:
    #     delta_R = None

    # Assuming 'action' column exists for entropy calculation
    # if 'action' in df.columns:
    #     action_counts = df['action'].value_counts(normalize=True)
    #     entropy = -(action_counts * np.log(action_counts)).sum()
    # else:
    #     entropy = None

    return {
        'file_name': os.path.basename(file_path),  # Capture the file name without the path
        'final_performance': final_performance,
        'avg_lane_deviation': avg_lane_deviation,
        # 'steering_variance': steering_variance,
        # 'steering_penalty': steering_penalty,
        # 'delta_R': delta_R,
        # 'entropy': entropy
    }

def process_files(directory,output_directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    results = {
        'file_name': [],
        'final_performance': [],
        'avg_lane_deviation': [],
        # 'steering_variance': [],
        # 'steering_penalty': [],
        # 'delta_R': [],
        # 'entropy': []
    }

    for file in csv_files:
        metrics = calculate_metrics(file)
        for key in results:
            if metrics[key] is not None:
                results[key].append(metrics[key])

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Writing results to separate files in the specified output directory
    for metric, values in results.items():
        if values:
            # Prepare DataFrame with file names and metric values
            data = pd.DataFrame({
                'File Name': results['file_name'],
                metric: results[metric]
            })
            output_path = os.path.join(output_directory, f'{metric}_results.csv')
            pd.DataFrame({metric: values}).to_csv(output_path, index=False)
            print(f"Saved {output_path}")

# Example usage
input_directory = '/Users/marziehghayour/Library/CloudStorage/GoogleDrive-marzieh.ghayour.na@gmail.com/My Drive/U of A/ECE 720/Project/Simulation/Map2/controllers/city/logs_new'
output_directory = '/Users/marziehghayour/Library/CloudStorage/GoogleDrive-marzieh.ghayour.na@gmail.com/My Drive/U of A/ECE 720/Project/Simulation/Map2/controllers/city/logs_new/results'  # Change this to your desired output directory
process_files(input_directory, output_directory)

#     # Writing results to separate files
#     for metric, values in results.items():
#         if values:
#             pd.DataFrame({metric: values}).to_csv(f'{metric}_results.csv', index=False)

# # Example usage
# process_files('/Users/marziehghayour/Library/CloudStorage/GoogleDrive-marzieh.ghayour.na@gmail.com/My Drive/U of A/ECE 720/Project/Simulation/Map2/controllers/city/logs_new')