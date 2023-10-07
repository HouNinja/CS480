import numpy as np 
import pandas as pd
import sys
from sklearn.linear_model import Ridge

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
    w_0 = np.zeros((column, 1))
    b_0 = 0
    vector_one = np.ones((row, 1))
    direction = np.zeros((column, 1))
    step_size = 0.4
    for _ in range(num_iterations):
        prev_w_0 = w_0
        direction = 1 / row * (X.T.dot(X.dot(w_0) + b_0 * vector_one - y) + 2 * lambda_value * direction)
        b_0 = b_0 - step_size * np.sum(X.dot(w_0) + b_0 * vector_one - y) / row
        w_0 = w_0 - step_size * direction
        if np.linalg.norm(w_0 - prev_w_0) <= 0.0001:
            break
    return w_0, b_0

def standardizing_data(X):
    return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

def Compute_error(X, w, b, y):
    row, column = X.shape
    vector = X.dot(w) + b * np.ones((row, 1)) - y
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
    y = Generating_flag_value_vector("housing_y_train.csv")
    X = standardizing_data(X)
    y = standardizing_data(y)
    lambda_value = 0

    X_test = Generating_Feature_Vector_Matrix("housing_X_test.csv")
    y_test = Generating_flag_value_vector("housing_y_test.csv")

    result_0 = Ridge_Regression(X, y, lambda_value)
    w = result_0[:-1]
    b = result_0[-1]
    Training_Error_Ridge = Compute_error(X, w, b, y)
    Training_loss_Ridge = Compute_Loss(X, w, b, y, lambda_value)
    Test_Error = Compute_error(X_test, w, b, y_test)

    ridge_model = Ridge(alpha = lambda_value)
    ridge_model.fit(X, y)
    y_pred = ridge_model.predict(X_test)
    y_diff = y_pred - y_test
    y_diff = y_diff.flatten()
    r, c = X_test.shape
    # print (1 / 2 / r * y_diff.dot(y_diff))


    w_1, b_1 = Ridge_Regression_Gradient_Descent(X, y, lambda_value, 20000000000)
    Training_Error_Ridge_1 = Compute_error(X, w_1, b_1, y)
    Training_loss_Ridge_1 = Compute_Loss(X, w_1, b_1, y, lambda_value)
    Test_Error_1 = Compute_error(X_test, w_1, b_1, y_test)
    print(Training_Error_Ridge)
    print(Training_loss_Ridge)
    print(Test_Error)

    print(Training_Error_Ridge_1)
    print(Training_loss_Ridge_1)
    print(Test_Error_1)
    