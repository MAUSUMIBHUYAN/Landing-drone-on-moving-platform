import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(filename='robot_trajectory_low_noise_log.csv'):
    """
    Loads trajectory data from the CSV log file and plots:
    1. Actual vs. Estimated Robot Path (XY plane).
    2. The 2D Position Estimation Error over time.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Log file '{filename}' not found.")
        print("Please ensure the ROS node ran successfully and created this file.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Log file '{filename}' is empty.")
        return

    print(f"Loaded {len(df)} data points from {filename}.")
    
    # --------------------------------------------------------
    # 1. Plot the XY Trajectory Comparison
    # --------------------------------------------------------
    plt.figure(figsize=(10, 8))

    # Convert Pandas Series to NumPy array using .values before plotting
    plt.plot(df['actual_x'].values, df['actual_y'].values, # <-- ADDED .values
             label='Actual Robot Path (Odometry)', color='blue', linewidth=2, alpha=0.8)
    
    # Convert Pandas Series to NumPy array using .values before plotting
    plt.plot(df['estimated_x'].values, df['estimated_y'].values, # <-- ADDED .values
             label='Estimated Robot Path (Kalman Filter)', color='red', linestyle='--', linewidth=2)

    # Convert Pandas Series to NumPy array using .values before plotting
    plt.plot(df['drone_x'].values, df['drone_y'].values, # <-- ADDED .values
             label='Drone Position', color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.title('Actual vs. Estimated Robot Trajectory', fontsize=16)
    plt.xlabel('X Position (m)', fontsize=14)
    plt.ylabel('Y Position (m)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') # Essential for a correct spatial representation
    plt.show()

    # --------------------------------------------------------
    # 2. Plot Error Over Time
    # --------------------------------------------------------
    
    # Calculate the estimation error
    error_x = df['actual_x'] - df['estimated_x']
    error_y = df['actual_y'] - df['estimated_y']
    distance_error = np.hypot(error_x, error_y)
    
    plt.figure(figsize=(12, 5))
    # Convert Pandas Series to NumPy array using .values before plotting
    plt.plot(df['time'].values, distance_error.values, label='2D Position Error (m)', color='purple') # <-- ADDED .values
    plt.title('Kalman Filter Position Estimation Error Over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Error Distance (m)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    plot_trajectories()