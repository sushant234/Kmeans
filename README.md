# K-Means Clustering with Mini-Batch and Restarts

This project implements a K-Means clustering algorithm with functionality for mini-batch optimization and multiple restarts. It allows the user to input a dataset, configure parameters like the number of centroids (`k`), iterations, and restarts, and visualizes the clustering results.

---

## Features
- **K-Means Clustering**: A robust implementation of the k-means algorithm.
- **Mini-Batch Optimization**: Efficient handling of large datasets using mini-batches.
- **Multiple Restarts**: Ensures the algorithm finds the best solution by running multiple restarts.
- **Error Handling**: Comprehensive error handling to manage missing files and invalid inputs.
- **Interactive Input**: Accepts dataset file path, `k`, number of iterations, and restarts as input.
- **Visualization**: Plots distortion costs for different restart numbers.

---

## Requirements
The script requires Python 3.7 or higher. Install the dependencies using the provided `requirements.txt` file:

```
pip3 install -r requirements.txt
```

### Setup for running PartA.py
1. Clone this repository or download the script.
2. Place your dataset in the same directory as the script. Ensure the dataset is in CSV format.
3. Run the script:
     ```
     python PartA.py
     ```
4. Enter the required inputs:
     1. File path to the dataset (e.g., clusteringData.csv)
     2. Number of centroids (k)
     3. Number of iterations
     4. Number of restarts
   The script will process the data and display a plot of distortion costs vs. restart numbers.

### Setup for running PartA.py
1. Clone this repository or download the script.
2. Place your dataset in the same directory as the script. Ensure the dataset is in CSV format.
3. Run the script:
     ```
     python PartA.py
     ```
4. Enter the required inputs:
     1. File path to the dataset (e.g., clusteringData.csv)
     2. Number of centroids (k)
     3. Number of iterations
     4. Number of restarts
5. The script will process the data and display a plot of distortion costs vs. restart numbers.

### Input Example

  ```
    Enter the file path of your dataset: clusteringData.csv
    Enter the number of centroids (k): 3
    Enter the number of iterations: 10
    Enter the number of restarts: 5
  ```

### Output

  A plot of distortion costs across multiple restarts to help you choose the best clustering result.
  Console output showing progress and final distortion cost.

### Dataset Format

  The dataset should be in CSV format with numerical values. Each row represents a data point, and each column       represents a feature.

  ```
    1.2,2.3,3.4
    4.5,5.6,6.7
    7.8,8.9,9.0
  ```

### Error Handling

  If the dataset file is not found, the script will notify the user.
  Invalid inputs will prompt the user to re-enter values.

### License

  This project is licensed under the MIT License.

### Author

  Sushant Anand Sarvade


