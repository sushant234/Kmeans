from turtle import distance
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from test import calculate_accuracy


# Function description :- The function returns the centroid from the feature_data
# input :- feature_data, hyperparameter_k
# output :- array of centroids
def initalize_centroids(feature_data, k):
    res = random.sample(range(0, len(feature_data)), k)
    a = feature_data[res]
    print('initial centroid: ',a)
    return a

# Function description :- The function returns the mini_batch points from the feature_data
# input :- feature_data, hyperparameter_k
# output :- array of mini_batch
def mini_batch(feature_data, mini_batch_size):
    # mini_batch_size random sample from length of feature
    res = random.sample(range(0, len(feature_data)), mini_batch_size)
    # getting center from feature data using res as a index
    a = feature_data[res]
    print('mini_batch: ',a)
    return a

# Function description :- Assign centroid to feature_data using current array of centroid
# input :- feature_data, current_array_of_centroids
# output :- centroid data assigned to each individual, distortion cost
def assign_centroids(feature_data, current_array_of_centroids):

    distance_calculated = []

    for i in current_array_of_centroids:
        a = ((abs(np.subtract(feature_data, i)))**2)
        c = np.sum(a, axis=1)
        d = np.sqrt(c)
        print('testing formula : ',a)
        print('testing length: ', len(a))
        print('value from formula: ', len(d))
        distance_calculated.append(d)

    print('distance calculated: ',distance_calculated)
    print('distance calculated: ',len(distance_calculated[0]))

    
    centroid_data_assigned_to_each_individual = np.argsort(distance_calculated, axis=0)[:1]

    distortion_cost = calculate_cost(feature_data, centroid_data_assigned_to_each_individual[0], current_array_of_centroids)
    return centroid_data_assigned_to_each_individual[0], distortion_cost

# Function description :- The function will calculate the distortion cost of the centroids
# input :- feature_data, current_array_of_centroids, centroid_data_assigned_to_each_individual
# output :- distortion cost
def calculate_cost(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids):
    unique, counts = np.unique(centroid_data_assigned_to_each_individual, return_index=True)
    dataset = np.array(centroid_data_assigned_to_each_individual)
    distortion_cost = 0
    for i in unique:
        a = np.where(dataset == i)
        aa = a[0]
        b = feature_data[aa, ]
        cc = (np.sum((abs(np.subtract(b, current_array_of_centroids[i])))**2))
        distortion_cost += cc
    print('hey distortion cost here ;)',distortion_cost)
    return distortion_cost



# Function description :- The function will generate new centroids for feature dataset
# input :- feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids
# output :- new_array_of_centroids
def move_centroids(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids):
    unique, counts = np.unique(centroid_data_assigned_to_each_individual, return_index=True)
    print('centroid data assigned to each individual: ', centroid_data_assigned_to_each_individual)
    
    # dict to store nnew cemtroids
    CC = {}

    countt = []
    for i in range(0,len(unique)):
        countt.append(0)

    # dict to store cluster and there counts
    v=dict(zip(unique, countt))
    
    print('current array of centroids: ', current_array_of_centroids)

   
    for x in range(0, len(feature_data)):
        # getting center of this x
        c = centroid_data_assigned_to_each_individual[x]
        # updating per center count
        v[c] = v[c]+1
        # getting per center learning rate
        learning_rate = (1/v[c])
        # taking gradient steps
        CC[int(c)] = (1-learning_rate)*c + (learning_rate*(feature_data[x]))

    a = list(CC.values())
    print('a aa: ',a[0])
    return a[0]


# Function description :- The function restarts, restart the kmeans with k different values and return the best solution found over the 10 
# restarts
# input :- feature_data,number_of_centroids, no_of_iterations, no_of_restarts
# output :- Best solution found over 10 restarts, k
def restart_kmeans(feature_data, no_of_iterations, no_of_restarts):
    current_array_of_centroids = []
    distortion_cost_fn_value1 = []
    centroids1 = []
    array_of_centroids1 = []
    k = []
    for i in range(0, no_of_restarts):
        k.append(i+1)
        current_array_of_centroids = initalize_centroids(feature_data, i+1)
        print('current array of centroids: ', current_array_of_centroids)
        distortion_cost_fn_value = []
        centroids = []
        array_of_centroids = []
        array_of_centroids.append(current_array_of_centroids)
        for i in range(0, no_of_iterations):
            centroid_data_assigned_to_each_individual,distortion_cost = assign_centroids(feature_data, current_array_of_centroids)
            # distortion_cost = calculate_cost(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids)
            centroids.append(centroid_data_assigned_to_each_individual)
            current_array_of_centroids = move_centroids(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids)
            array_of_centroids.append(current_array_of_centroids)
            distortion_cost_fn_value.append(distortion_cost)

        print('best distortion values: ',np.argsort(distortion_cost_fn_value)[:1])
        abc = np.argsort(distortion_cost_fn_value)[:1]
        abc1 = abc[0]
        distortion_cost_fn_value1.append(distortion_cost_fn_value[abc1])
        centroids1.append(centroids[abc1])
        array_of_centroids1.append(array_of_centroids[abc1])


    abc = np.argsort(distortion_cost_fn_value1)[:1]
    abc1 = abc[0]
    print('distortion cost function :',distortion_cost_fn_value1[abc1])
    print('distortion cost function value :',distortion_cost_fn_value1)
  

    print('array of centroids', array_of_centroids1[abc1])
    return distortion_cost_fn_value1, k



def main():
    feature_dataset = np.loadtxt("clusteringData.csv", delimiter=',')
    feature_data = feature_dataset[:, ]
    print(feature_dataset[:, ])

    mini_batch_size = len(feature_data)/7
    no_of_iterations=10
    no_of_restarts = 10
    size_of_mini_batch = 80000
    mini_batch_array = mini_batch(feature_data, size_of_mini_batch)
    print('mini batch array: ', mini_batch_array)
    a,b = restart_kmeans(mini_batch_array, no_of_iterations, no_of_restarts)
    xpoints = np.array(b)
    ypoints = np.array(a)
    plt.plot(xpoints, ypoints)
    plt.show()
    exit()

if __name__ == '__main__':
    main()