import numpy as np 
import pandas as pd
import sys

def Generating_Feature_Vector_Matrix(filename: str):
    data = pd.read_csv(filename, header = None)
    matrix = data.T.values
    return matrix

def Generating_flag_value_vector(filename: str):
    data = pd.read_csv(filename, header = None)
    vector = data.values
    return vector

def Ridge_Regression(X, y, lambda_value):
    row, column = X.shape
    new_column = np.ones(row)
    padding_matrix_one = np.column_stack((X, new_column))
    padding_Trans_matrix_one = padding_matrix_one.T
    identity_matrix = np.identity(column + 1)
    return np.linalg.solve(padding_Trans_matrix_one.dot(padding_matrix_one) + 2 * lambda_value * row * identity_matrix, padding_Trans_matrix_one.dot(y))

def Ridge_Regression_Gradient_Descent(X, y, lambda_value, num_iterations):
    row, column = X.shape
    w_0 = np.zeros(column)
    b_0 = 0
    vector_one = np.ones(row)
    direction = np.zeros(column)
    step_size = 1.0
    for _ in range(num_iterations):
        prev_direction = direction
        direction = 1 / row * X.T.dot(X.dot(direction) + b * vector_one - y) + 2 * lambda_value * direction
        w_0 = w_0 - step_size * direction
        b_0 = b_0 - step_size * np.sum(X.dot(direction) + b * vector_one - y)
        if np.linalg.norm(direction - prev_direction) <= 0.0001:
            break
    w_0.append(b_0)
    return w_0

def standardizing_data(X):
    return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

def Compute_error(X, w, b, y):
    row, column = X.shape
    vector = X.dot(w) + b * np.ones(row) - y
    vector = vector.flatten()
    return 1 / 2 / row * vector.dot(vector)

def Compute_Loss(X, w, b, y, lambda_value):
    row, column = X.shape
    vector = X.dot(w) + b * np.ones(row) - y
    vector = vector.flatten()
    flattened_w = w.flatten()
    return 1 / 2 / row * vector.T.dot(vector) + lambda_value * flattened_w.dot(flattened_w)


if __name__ == "__main__":
    X = Generating_Feature_Vector_Matrix("housing_X_train.csv")
    X = standardizing_data(X)
    y = Generating_flag_value_vector("housing_y_train.csv")
    y = standardizing_data(y)

    X_test = Generating_Feature_Vector_Matrix("housing_X_test.csv")
    X_test = standardizing_data(X)
    y_test = Generating_flag_value_vector("housing_y_test.csv")
    y_test = standardizing_data(y)

    result_0 = Ridge_Regression(X, y, 0)
    w = result_0[:-1]
    b = result_0[-1]
    print(b)
    Training_Error_Ridge = Compute_error(X, w, b, y)
    Training_loss_Ridge = Compute_Loss(X, w, b, y, 0)

    print(Training_Error_Ridge)
    print(Training_loss_Ridge)