import numpy as np
import random
import matplotlib.pyplot as plt

# Function description: Generate initial centroids randomly from feature data
def generate_centroids(feature_data, k):
    try:
        res = random.sample(range(0, len(feature_data)), k)
        return feature_data[res]
    except Exception as e:
        print(f"Error in generate_centroids: {e}")
        return []

# Function description: Assign feature data points to the nearest centroid
def assign_centroids(feature_data, current_array_of_centroids):
    try:
        distance_calculated = []
        for centroid in current_array_of_centroids:
            distances = np.linalg.norm(feature_data - centroid, axis=1)
            distance_calculated.append(distances)

        distance_calculated = np.array(distance_calculated)
        assigned_centroids = np.argmin(distance_calculated, axis=0)

        distortion_cost = calculate_cost(feature_data, assigned_centroids, current_array_of_centroids)
        return assigned_centroids, distortion_cost
    except Exception as e:
        print(f"Error in assign_centroids: {e}")
        return None, None

# Function description: Update centroids based on the mean of assigned points
def move_centroids(feature_data, assigned_centroids, current_array_of_centroids):
    try:
        new_centroids = []
        for cluster in range(len(current_array_of_centroids)):
            cluster_points = feature_data[assigned_centroids == cluster]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(current_array_of_centroids[cluster])
        return np.array(new_centroids)
    except Exception as e:
        print(f"Error in move_centroids: {e}")
        return current_array_of_centroids

# Function description: Calculate the distortion cost
def calculate_cost(feature_data, assigned_centroids, centroids):
    try:
        total_cost = 0
        for cluster in range(len(centroids)):
            cluster_points = feature_data[assigned_centroids == cluster]
            total_cost += np.sum(np.square(cluster_points - centroids[cluster]))
        return total_cost
    except Exception as e:
        print(f"Error in calculate_cost: {e}")
        return float('inf')

# Function description: Perform multiple restarts of k-means and return the best solution
def restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts):
    try:
        best_cost = float('inf')
        best_centroids = None
        all_costs = []

        for restart in range(no_of_restarts):
            print(f"Restart {restart + 1}/{no_of_restarts}")
            current_centroids = generate_centroids(feature_data, k)
            for iteration in range(no_of_iterations):
                assigned_centroids, cost = assign_centroids(feature_data, current_centroids)
                if assigned_centroids is None:
                    break
                current_centroids = move_centroids(feature_data, assigned_centroids, current_centroids)

            if cost < best_cost:
                best_cost = cost
                best_centroids = current_centroids

            all_costs.append(best_cost)

        return all_costs, list(range(1, no_of_restarts + 1))
    except Exception as e:
        print(f"Error in restart_kmeans: {e}")
        return [], []

# Function description: Get user input for the dataset and parameters
def get_user_input():
    try:
        dataset_path = input("Enter the dataset file path (e.g., 'clusteringData.csv'): ").strip()
        k = int(input("Enter the number of centroids (k): "))
        no_of_iterations = int(input("Enter the number of iterations: "))
        no_of_restarts = int(input("Enter the number of restarts: "))
        return dataset_path, k, no_of_iterations, no_of_restarts
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter numeric values for k, iterations, and restarts.")
        return None, None, None, None

# Main function
def main():
    try:
        # Get user inputs
        dataset_path, k, no_of_iterations, no_of_restarts = get_user_input()
        if not dataset_path or k is None or no_of_iterations is None or no_of_restarts is None:
            print("Exiting due to invalid inputs.")
            return

        # Load dataset
        try:
            feature_dataset = np.loadtxt(dataset_path, delimiter=',')
            feature_data = feature_dataset[:, ]
            print("Dataset loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File '{dataset_path}' not found.")
            return
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Perform k-means clustering with restarts
        costs, restarts = restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts)

        # Plot results
        if costs and restarts:
            plt.plot(restarts, costs, marker='o')
            plt.xlabel('Restart Number')
            plt.ylabel('Distortion Cost')
            plt.title('K-Means Clustering Distortion Costs')
            plt.grid(True)
            plt.show()
        else:
            print("No valid results to plot.")

    except Exception as e:
        print(f"Unexpected error in main: {e}")

if __name__ == '__main__':
    main()
