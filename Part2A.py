import numpy as np
import random
import matplotlib.pyplot as plt

# Function description: The function returns the centroid from the feature_data
def generate_centroids(feature_data, k):
    try:
        res = random.sample(range(0, len(feature_data)), k)
        array_of_centroid = feature_data[res]
        return array_of_centroid
    except Exception as e:
        print(f"Error in generate_centroids: {e}")
        return []

# Function description: Assign centroid to feature_data using the current array of centroids
def assign_centroids(feature_data, current_array_of_centroids):
    try:
        distance_calculated = []
        for i in current_array_of_centroids:
            distance_calculated_a = ((abs(np.subtract(feature_data, i)))**1)
            distance_calculated_c = np.sum(distance_calculated_a, axis=1)
            distance_calculated_d = distance_calculated_c**(1/1)
            distance_calculated.append(distance_calculated_d)

        centroid_data_assigned_to_each_individual = np.argsort(distance_calculated, axis=0)[:1]
        distortion_cost = calculate_cost(feature_data, centroid_data_assigned_to_each_individual[0], current_array_of_centroids)
        return centroid_data_assigned_to_each_individual[0], distortion_cost
    except Exception as e:
        print(f"Error in assign_centroids: {e}")
        return None, None

# Function description: Generate new centroids for the feature dataset
def move_centroids(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids):
    try:
        unique, counts = np.unique(centroid_data_assigned_to_each_individual, return_index=True)
        dataset = np.array(centroid_data_assigned_to_each_individual)
        new_array_of_centroids = []

        for i in unique:
            a = np.where(dataset == i)
            aa = a[0]
            b = feature_data[aa, ]
            c = b.mean(axis=0)
            new_array_of_centroids.append(c)
        return new_array_of_centroids
    except Exception as e:
        print(f"Error in move_centroids: {e}")
        return []

# Function description: Calculate the distortion cost of the centroids
def calculate_cost(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids):
    try:
        unique, counts = np.unique(centroid_data_assigned_to_each_individual, return_index=True)
        dataset = np.array(centroid_data_assigned_to_each_individual)
        distortion_cost = 0
        for i in unique:
            a = np.where(dataset == i)
            aa = a[0]
            b = feature_data[aa, ]
            cc = (np.sum((abs(np.subtract(b, current_array_of_centroids[i])))**2))
            distortion_cost += cc
        return distortion_cost
    except Exception as e:
        print(f"Error in calculate_cost: {e}")
        return float('inf')

# Function description: Restart k-means with multiple restarts and return the best solution
def restart_kmeans(feature_data, number_of_centroids, no_of_iterations, no_of_restarts):
    try:
        distortion_cost_fn_value1 = []
        centroids1 = []
        array_of_centroids1 = []
        k = []
        for i in range(no_of_restarts):
            k.append(i + 1)
            current_array_of_centroids = generate_centroids(feature_data, i + 1)
            distortion_cost_fn_value = []
            centroids = []
            array_of_centroids = []
            array_of_centroids.append(current_array_of_centroids)

            for j in range(no_of_iterations):
                centroid_data_assigned_to_each_individual, distortion_cost = assign_centroids(feature_data, current_array_of_centroids)
                if centroid_data_assigned_to_each_individual is None:
                    break
                centroids.append(centroid_data_assigned_to_each_individual)
                current_array_of_centroids = move_centroids(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids)
                array_of_centroids.append(current_array_of_centroids)
                distortion_cost_fn_value.append(distortion_cost)

            if distortion_cost_fn_value:
                abc = np.argsort(distortion_cost_fn_value)[:1]
                abc1 = abc[0]
                distortion_cost_fn_value1.append(distortion_cost_fn_value[abc1])
                centroids1.append(centroids[abc1])
                array_of_centroids1.append(array_of_centroids[abc1])

        abc = np.argsort(distortion_cost_fn_value1)[:1]
        abc1 = abc[0]
        print('Best distortion cost function:', distortion_cost_fn_value1[abc1])
        return distortion_cost_fn_value1, k
    except Exception as e:
        print(f"Error in restart_kmeans: {e}")
        return [], []

# Main execution flow
def main():
    try:
        feature_dataset = np.loadtxt("clusteringData.csv", delimiter=',')
        feature_data = feature_dataset[:, ]
        k = 3
        no_of_iterations = 10
        no_of_restarts = 10

        a, b = restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts)
        if a and b:
            xpoints = np.array(b)
            ypoints = np.array(a)
            plt.plot(xpoints, ypoints)
            plt.xlabel('Restarts')
            plt.ylabel('Distortion Cost')
            plt.title('K-Means Clustering Distortion Costs')
            plt.show()
    except FileNotFoundError:
        print("Error: File 'clusteringData.csv' not found.")
    except Exception as e:
        print(f"Unexpected error in main: {e}")

if __name__ == '__main__':
    main()
