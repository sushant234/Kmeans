import numpy as np
import random
import matplotlib.pyplot as plt

# Function description: Initialize centroids randomly from the feature data
def initialize_centroids(feature_data, k):
    try:
        res = random.sample(range(0, len(feature_data)), k)
        centroids = feature_data[res]
        print('Initial centroids:', centroids)
        return centroids
    except Exception as e:
        print(f"Error in initialize_centroids: {e}")
        return None

# Function description: Generate a mini-batch of points from the feature data
def mini_batch(feature_data, mini_batch_size):
    try:
        res = random.sample(range(0, len(feature_data)), mini_batch_size)
        batch = feature_data[res]
        print('Mini-batch:', batch)
        return batch
    except Exception as e:
        print(f"Error in mini_batch: {e}")
        return None

# Function description: Assign centroid to feature data points
def assign_centroids(feature_data, current_array_of_centroids):
    try:
        distances = []

        for centroid in current_array_of_centroids:
            diff = np.subtract(feature_data, centroid)
            squared_diff = np.square(diff)
            distance = np.sqrt(np.sum(squared_diff, axis=1))
            distances.append(distance)

        distances = np.array(distances)
        assigned_centroids = np.argmin(distances, axis=0)

        distortion_cost = calculate_cost(feature_data, assigned_centroids, current_array_of_centroids)
        return assigned_centroids, distortion_cost
    except Exception as e:
        print(f"Error in assign_centroids: {e}")
        return None, None

# Function description: Calculate the distortion cost
def calculate_cost(feature_data, assigned_centroids, centroids):
    try:
        distortion_cost = 0
        for cluster_index in range(len(centroids)):
            cluster_points = feature_data[assigned_centroids == cluster_index]
            centroid = centroids[cluster_index]
            cluster_distances = np.sum(np.square(cluster_points - centroid))
            distortion_cost += cluster_distances
        print('Distortion cost:', distortion_cost)
        return distortion_cost
    except Exception as e:
        print(f"Error in calculate_cost: {e}")
        return float('inf')

# Function description: Move centroids using a learning rate approach
def move_centroids(feature_data, assigned_centroids, centroids):
    try:
        unique_clusters = np.unique(assigned_centroids)
        updated_centroids = np.copy(centroids)
        counts = {cluster: 0 for cluster in unique_clusters}

        for i, cluster in enumerate(assigned_centroids):
            counts[cluster] += 1
            learning_rate = 1 / counts[cluster]
            updated_centroids[cluster] = (1 - learning_rate) * updated_centroids[cluster] + learning_rate * feature_data[i]

        print('Updated centroids:', updated_centroids)
        return updated_centroids
    except Exception as e:
        print(f"Error in move_centroids: {e}")
        return centroids

# Function description: Restart k-means clustering multiple times and find the best solution
def restart_kmeans(feature_data, no_of_iterations, no_of_restarts):
    try:
        best_cost = float('inf')
        best_centroids = None
        all_costs = []
        k_values = []

        for restart in range(1, no_of_restarts + 1):
            print(f"Restart {restart}")
            current_centroids = initialize_centroids(feature_data, restart)
            if current_centroids is None:
                continue

            for iteration in range(no_of_iterations):
                assigned_centroids, cost = assign_centroids(feature_data, current_centroids)
                if assigned_centroids is None:
                    break

                current_centroids = move_centroids(feature_data, assigned_centroids, current_centroids)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_centroids = current_centroids

            k_values.append(restart)

        print('Best distortion cost:', best_cost)
        return all_costs, k_values
    except Exception as e:
        print(f"Error in restart_kmeans: {e}")
        return [], []

# Function description: Get user input from the terminal
def get_user_input():
    try:
        dataset_path = input("Enter the dataset file path (e.g., 'clusteringData.csv'): ").strip()
        mini_batch_size = int(input("Enter the mini-batch size: "))
        no_of_iterations = int(input("Enter the number of iterations: "))
        no_of_restarts = int(input("Enter the number of restarts: "))
        return dataset_path, mini_batch_size, no_of_iterations, no_of_restarts
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter numeric values for batch size, iterations, and restarts.")
        return None, None, None, None

# Main function
def main():
    try:
        # Get inputs from user
        dataset_path, mini_batch_size, no_of_iterations, no_of_restarts = get_user_input()
        if not dataset_path or not mini_batch_size or not no_of_iterations or not no_of_restarts:
            print("Exiting due to invalid inputs.")
            return

        # Load dataset
        try:
            feature_dataset = np.loadtxt(dataset_path, delimiter=',')
            feature_data = feature_dataset[:, ]
            print('Feature data loaded.')
        except FileNotFoundError:
            print(f"Error: File '{dataset_path}' not found.")
            return

        # Create mini-batch and run k-means
        mini_batch_array = mini_batch(feature_data, mini_batch_size)
        if mini_batch_array is None:
            return

        print('Mini-batch created.')
        costs, k_values = restart_kmeans(mini_batch_array, no_of_iterations, no_of_restarts)

        # Plot results
        if costs and k_values:
            plt.plot(k_values, costs)
            plt.xlabel('Restarts')
            plt.ylabel('Distortion Cost')
            plt.title('K-Means Distortion Costs')
            plt.show()

    except Exception as e:
        print(f"Unexpected error in main: {e}")

if __name__ == '__main__':
    main()
