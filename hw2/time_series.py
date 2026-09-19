import numpy as np
import copy
import time
import math

# You may use your own arguments and return values depending on your implementation
def load_split_data():
    '''
    Load temperature.csv and split data
    '''
    return

# You may use your own arguments and return values depending on your implementation.
def create_matrices():
    '''
    Create data matrices
    '''
    return

# (3.2.1, 3.2.2) OLS estimation
# This function will be autograded, please use the same arguments and return values as specified.
def OLS(D, X_train, y_train, X_test, y_test):
    '''
    Arguments:
        D: number of timesteps
        X_train: 2D numpy array of size ((T-D), D)
        y_train: 1D numpy array of length (T-D)
        X_test: 2D numpy array of size ((C-D), D)
        y_test: 1D numpy array of length (C-D)
    Returns:
        ols_weights: List (the first min(10, D) feature weights, followed by
            the bias weight as the final element; 11 elements when D >= 10)
        ols_time: Float (number of seconds taken to fit OLS)
        ols_test_mse: Float (MSE of model on test data)
    '''
    return

# (3.2.4) SGD estimation
# This function will be autograded, please use the same arguments and return values as specified.
def SGD(D, X_train, y_train, X_test, y_test, num_epochs=20, lr=1e-10):
    '''
    Arguments:
        D: number of timesteps
        X_train: 2D numpy array of size ((T-D), D)
        y_train: 1D numpy array of length (T-D)
        X_test: 2D numpy array of size ((C-D), D)
        y_test: 1D numpy array of length (C-D)
        num_epochs: Int (number of epochs)
        lr: Float (learning rate)
    Returns:
        sgd_train_mse: Float (MSE of model on train data)
        sgd_test_mse: Float (MSE of model on test data)
        sgd_time: Float (number of seconds taken to train SGD)
        sgd_weights: List (the first min(10, D) feature weights, followed by
            the bias weight as the final element; 11 elements when D >= 10)
    '''
    return

if __name__ == '__main__':
    # Load, split and preprocess the data
    train_data, test_data = load_split_data(...)

    # Set the number of timesteps
    D = 10

    # Create data matrices
    X_train, y_train = create_matrices(...)
    X_test, y_test = create_matrices(...)

    # OLS model
    ols_weights, ols_time, ols_test_mse = OLS(D, X_train, y_train, X_test, y_test)

    # SGD model
    sgd_train_mse, sgd_test_mse, sgd_time, sgd_weights = SGD(
        D, X_train, y_train, X_test, y_test, num_epochs=20, lr=1e-10
    )
