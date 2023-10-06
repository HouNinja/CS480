import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import sys


def Generating_Feature_Vector_Matrix(filename: str):
    data = pd.read_csv(filename, header = None)
    matrix = data.values
    return matrix

def Generating_flag_value_vector(filename: str):
    data = pd.read_csv(filename, header = None)
    vector = data.values
    return vector

def Split_data(X, k = 10):
    row, column = X.shape
    split_matrices = np.array_split(X, k)
    return split_matrices
    
def Ridge_Regression_CV(Split_data, Split_y, hyperparameters):
    best_performance = float("inf")
    output = -1
    for i in hyperparameters:
        ridge_model = Ridge(alpha = i)
        performance = 0
        row, column = Split_data[0].shape
        for index in range(len(Split_data)):
            data = np.zeros((0,column))
            real_y = np.zeros((0,1))
            for t in range(len(Split_data)):
                if t != index:
                    data = np.vstack((data, Split_data[t]))
                    real_y = np.vstack((real_y, Split_y[t]))
            ridge_model.fit(data, real_y)
            predicted_y = ridge_model.predict(Split_data[index])
            difference_y = Split_y[index] - predicted_y
            difference_y = difference_y.flatten()
            performance += difference_y.dot(difference_y)
        if performance < best_performance:
            output = i
    return output

def Lasso_CV(Split_data, Split_y, hyperparameters):
    best_performance = float("inf")
    output = -1
    for i in hyperparameters:
        lasso_model = Lasso(alpha = i)
        performance = 0
        row, column = Split_data[0].shape
        for index in range(len(Split_data)):
            data = np.zeros((0,column))
            real_y = np.zeros((0,1))
            for t in range(len(Split_data)):
                if t != index:
                    data = np.vstack((data, Split_data[t]))
                    real_y = np.vstack((real_y, Split_y[t]))
            lasso_model.fit(data, real_y)
            predicted_y = lasso_model.predict(Split_data[index])
            difference_y = Split_y[index] - predicted_y
            difference_y = difference_y.flatten()
            performance += difference_y.dot(difference_y)
        if performance < best_performance:
            output = i
    return output

def Ridge_Compute_Error(X, y, hyperparameter):
    ridge_model = Ridge(alpha = i)
    ridge_model.fit(X, y)
    y_pred = ridge_model.predict(X)

def Plotting_Histogram(X, y, X_test, y_test, hyperparameters):
    linear_model = Ridge(alpha = 0)
    linear_model.fit(X, y)
    linear_coef = linear_model.coef_.flatten()
    y_diff = linear_model.predict(X_test) - y_test
    y_diff = y_diff.flatten()
    linear_error = y_diff.dot(y_diff)
    data = []
    data.append(linear_coef)

    ridge_model = Ridge(alpha = hyperparameters[0])
    ridge_model.fit(X,y)
    ridge_coef = ridge_model.coef_.flatten()
    y_diff = ridge_model.predict(X_test) - y_test
    y_diff = y_diff.flatten()
    ridge_error = y_diff.dot(y_diff)
    data.append(ridge_coef)

    lasso_model = Lasso(alpha = hyperparameters[1])
    lasso_model.fit(X,y)
    lasso_coef = lasso_model.coef_.flatten()
    y_diff = lasso_model.predict(X_test) - y_test
    y_diff = y_diff.flatten()
    lasso_error = y_diff.dot(y_diff)
    data.append(lasso_coef)

    range_min = min(np.min(linear_coef, axis = 0), np.min(ridge_coef, axis = 0), np.min(lasso_coef, axis = 0))
    range_max = max(np.max(linear_coef, axis = 0), np.max(ridge_coef, axis = 0), np.max(lasso_coef, axis = 0))

    bin_range = (range_min, range_max)
    num_bins = 60
    plt.hist(data, bins=num_bins, range=bin_range, color=['blue', 'red', 'pink'], edgecolor='black', label=[f"Linear Regression, err={linear_error}", f"Ridge Regression, reg={hyperparameters[0]},err={ridge_error}", f"Lasso, reg={hyperparameters[1]},err={lasso_error}"])

    plt.xlabel('Range of values in Parameter vector')
    plt.ylabel('Frequency')
    plt.title('Histogram with Custom Range and Bins')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    training_X_file_name = "X_train_" + sys.argv[1] + ".csv"
    training_Y_file_name = "Y_train_" + sys.argv[1] + ".csv"
    testing_X_file_name = "X_test_" + sys.argv[1] + ".csv"
    testing_Y_file_name = "Y_test_" + sys.argv[1] + ".csv"

    X = Generating_Feature_Vector_Matrix(training_X_file_name)
    y = Generating_flag_value_vector(training_Y_file_name)
    X_test = Generating_Feature_Vector_Matrix(testing_X_file_name)
    y_test = Generating_flag_value_vector(testing_Y_file_name)
    split_X = Split_data(X)
    split_y = Split_data(y)
    unregularized = [0]
    hyperparameters = [0.05, 0.1, 0.5, 1.0]
    selected_hyperparameters = []
    selected_hyperparameters.append(Ridge_Regression_CV(split_X, split_y, hyperparameters))
    selected_hyperparameters.append(Lasso_CV(split_X, split_y, hyperparameters))
    Plotting_Histogram(X, y, X_test, y_test, selected_hyperparameters)