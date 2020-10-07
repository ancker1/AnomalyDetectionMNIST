import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize the input data to inverval [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255
    # Flatten each image into a vecto'r
    x_train = x_train.reshape(len(x_train), 784)
    x_test  = x_test.reshape(len(x_test), 784)

    return (x_train, y_train), (x_test, y_test)

def create_dataset(x_test, y_test, normal=4, contamination=0.10):
    num_normals  = int(np.sum( y_test == normal ))
    num_outliers = int(np.ceil( num_normals * contamination ))
    # Get non-normal elements in test dataset
    sub_arr = x_test[ y_test != normal ]
    # Generate random non-normal indices
    idx = np.random.randint(low=0, high=len(sub_arr), size=num_outliers)
    # Create contaminated test data
    x_contaminated = np.zeros((num_normals + num_outliers, 784))
    x_contaminated[num_outliers:,:] = x_test[y_test == normal]
    x_contaminated[:num_outliers,:] = sub_arr[idx]
    # Create contaminated labels
    y_contaminated = np.zeros((num_normals + num_outliers))
    y_contaminated[num_outliers:] = normal
    y_contaminated[:num_outliers] = y_test[ y_test != normal ][idx]
    # Create binary anomaly labels,   normal = 0,   outlier = 1
    binary_labels = np.copy( y_contaminated )
    binary_labels[y_contaminated == normal] = 0
    binary_labels[y_contaminated != normal] = 1
    
    return (x_contaminated, y_contaminated, binary_labels)