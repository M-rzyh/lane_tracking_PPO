import pandas as pd
import matplotlib.pyplot as plt

# Loading data from different CSV files
data1 = pd.read_csv('model1.csv')
data2 = pd.read_csv('model2.csv')
data3 = pd.read_csv('model3.csv')


plt.figure(figsize=(12, 6))

# Plot each dataset with mean line and shaded standard deviation
plt.plot(data1['Steps'], data1['Average Return'], label='Model 1', color='blue')
plt.fill_between(data1['Steps'], data1['Average Return'] - data1['Standard Deviation'], data1['Average Return'] + data1['Standard Deviation'], color='blue', alpha=0.2)

plt.plot(data2['Steps'], data2['Average Return'], label='Model 2', color='red')
plt.fill_between(data2['Steps'], data2['Average Return'] - data2['Standard Deviation'], data2['Average Return'] + data2['Standard Deviation'], color='red', alpha=0.2)

plt.plot(data3['Steps'], data3['Average Return'], label='Model 3', color='green')
plt.fill_between(data3['Steps'], data3['Average Return'] - data3['Standard Deviation'], data3['Average Return'] + data3['Standard Deviation'], color='green', alpha=0.2)

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Average Return')
plt.title('Learning Curves of Different Models')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()