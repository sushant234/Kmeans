import numpy as np
import random
import matplotlib.pyplot as plt





# Function description :- The function returns the centroid from the feature_data
# input :- feature_data, hyperparameter_k
# output :- array of centroids
def generate_centroids(feature_data, k):
    # k random sample from length of feature
    res = random.sample(range(0, len(feature_data)), k)
    # getting centroid from feature data using res as a index
    array_of_centroid = feature_data[res]
    # return array of centroids
    return array_of_centroid



# Function description :- Assign centroid to feature_data using current array of centroid
# input :- feature_data, current_array_of_centroids
# output :- centroid data assigned to each individual, distortion cost
def assign_centroids(feature_data, current_array_of_centroids):
    distance_calculated = []
    # calculating distance for each centroids
    for i in current_array_of_centroids:
        distance_calculated_a = ((abs(np.subtract(feature_data, i)))**1)
        distance_calculated_c = np.sum(distance_calculated_a, axis=1)
        distance_calculated_d = distance_calculated_c**(1/1)
        print('testing formula : ',distance_calculated_a)
        print('testing length: ', len(distance_calculated_a))
        print('value from formula: ', len(distance_calculated_d))
        distance_calculated.append(distance_calculated_d)

    print('distance calculated: ',distance_calculated)
    print('distance calculated: ',len(distance_calculated[0]))

    # Sorting the distance of all the centroid with axis value 0, which return 1 index value for every points.
    centroid_data_assigned_to_each_individual = np.argsort(distance_calculated, axis=0)[:1]
    print('centroid_data_assigned_to_each_individual: ', len(centroid_data_assigned_to_each_individual[0]))
    # fucntion will return the distoriton cost function
    distortion_cost = calculate_cost(feature_data, centroid_data_assigned_to_each_individual[0], current_array_of_centroids)
    return centroid_data_assigned_to_each_individual[0], distortion_cost



# Function description :- The function will generate new centroids for feature dataset
# input :- feature_data, current_array_of_centroids, centroid_data_assigned_to_each_individual
# output :- new_array_of_centroids
def move_centroids(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids):
    # obtaining unique values present in the list
    unique, counts = np.unique(centroid_data_assigned_to_each_individual, return_index=True)
    dataset = np.array(centroid_data_assigned_to_each_individual)
    new_array_of_centroids = []

    for i in unique:
        # get index of i from the dataset
        a = np.where(dataset == i)
        aa = a[0]
        # obtain values of feature_data of that index
        b = feature_data[aa, ]
        # calculate the mean 
        c = b.mean(axis=0)
        # appending new centroids
        new_array_of_centroids.append(c)
    print('new array of centroids: ',new_array_of_centroids)
    return new_array_of_centroids


# Function description :- The function will calculate the distortion cost of the centroids
# input :- feature_data, current_array_of_centroids, centroid_data_assigned_to_each_individual
# output :- distortion cost
def calculate_cost(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids):
    # obtaining unique values present in the list
    unique, counts = np.unique(centroid_data_assigned_to_each_individual, return_index=True)
    dataset = np.array(centroid_data_assigned_to_each_individual)
    distortion_cost = 0
    for i in unique:
        # get index of i from the dataset
        a = np.where(dataset == i)
        aa = a[0]
         # obtain values of feature_data of that index
        b = feature_data[aa, ]
        # calculating distortion cost function, calcualting difference between b and centroid using
        # numpy subtract and using abs for postive values then squaring the equation and adding them all using numpy summation
        cc = (np.sum((abs(np.subtract(b, current_array_of_centroids[i])))**2))
        distortion_cost += cc
    print('hey distortion cost here ;)',distortion_cost)
    return distortion_cost


# Function description :- The function restarts, restart the kmeans with k different values and return the best solution found over the 10 
# restarts
# input :- feature_data,number_of_centroids, no_of_iterations, no_of_restarts
# output :- Best solution found over 10 restarts, k
def restart_kmeans(feature_data, number_of_centroids, no_of_iterations, no_of_restarts):
    current_array_of_centroids = []
    distortion_cost_fn_value1 = []
    centroids1 = []
    array_of_centroids1 = []
    k = []
    # restarting kmeans
    for i in range(0, no_of_restarts):
        k.append(i+1)
        # calling generate_centroid
        current_array_of_centroids = generate_centroids(feature_data, i+1)
        distortion_cost_fn_value = []
        centroids = []
        array_of_centroids = []
        array_of_centroids.append(current_array_of_centroids)
        # number of iterations
        for i in range(0, no_of_iterations):
            # calling assign_centroids
            centroid_data_assigned_to_each_individual,distortion_cost = assign_centroids(feature_data, current_array_of_centroids)
            centroids.append(centroid_data_assigned_to_each_individual)
            # calling move_centroids
            current_array_of_centroids = move_centroids(feature_data, centroid_data_assigned_to_each_individual, current_array_of_centroids)
            array_of_centroids.append(current_array_of_centroids)
            distortion_cost_fn_value.append(distortion_cost)

        print('best distortion values: ',np.argsort(distortion_cost_fn_value)[:1])
        # getting best distortion cost from one restart
        abc = np.argsort(distortion_cost_fn_value)[:1]
        abc1 = abc[0]
        # appending the best distotrion cost of every restart
        distortion_cost_fn_value1.append(distortion_cost_fn_value[abc1])
        centroids1.append(centroids[abc1])
        array_of_centroids1.append(array_of_centroids[abc1])

    # getting index of best distotrion cost from all the restarts
    abc = np.argsort(distortion_cost_fn_value1)[:1]
    abc1 = abc[0]
    print('distortion cost function :',distortion_cost_fn_value1[abc1])
    print(' :',distortion_cost_fn_value1)
  
    print('array of centroids', array_of_centroids1[abc1])
    return distortion_cost_fn_value1, k



def main():
    # reading the csv file
    feature_dataset = np.loadtxt("clusteringData.csv", delimiter=',')
    feature_data = feature_dataset[:, ]
    print(feature_dataset[:, ])
    k=3
    no_of_iterations=10
    no_of_restarts = 10
    # calling restaet_kmeans function
    a,b = restart_kmeans(feature_data, k, no_of_iterations, no_of_restarts)
    xpoints = np.array(b)
    ypoints = np.array(a)
    # ploting graph
    plt.plot(xpoints, ypoints)
    plt.show()





if __name__ == '__main__':
    main()