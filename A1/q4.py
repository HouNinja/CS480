import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from q2 import Ridge_Regression, Compute_error

def Generating_Feature_Vector_Matrix(filename: str):
    data = pd.read_csv(filename, header = None)
    matrix = data.values
    return matrix

def Generating_flag_value_vector(filename: str):
    data = pd.read_csv(filename, header = None)
    vector = data.values
    return vector

def Scatter_Plot(X, y, X_test, y_test, k):
    result = Ridge_Regression(X, y, 0)
    w = result[:-1]
    b = result[-1]
    row, column = X.shape
    linear_y = X_test.dot(w) + b * np.ones((row, 1))
    print(Compute_error(X_test, w, b, y_test))

    knn_errors = []
    knn_model = KNNRegression(X, y, k)

    y_pred = np.array(knn_model.predict(X_test))
    y_diff = y_test - y_pred
    y_diff = y_diff.flatten()

    row, column = X.shape
    plt.scatter(X_test, y_test, color = 'blue' , label=f"KNN algorithm with k={k}" )
    plt.scatter(X_test, linear_y, color = 'red', label=f"linear")
    plt.xlabel('X value')
    plt.ylabel('Y value')
    plt.legend()
    plt.show()

# def Plotting_Histogram(X, y, X_test, y_test):




class KNNRegression:
    def __init__(self, X, y, k):
        self.Training_X = X
        self.Training_y = y
        self.k = k
    
    def predict(self, X):
        output = []
        for x in X:
            distance = np.linalg.norm(self.Training_X - x, axis = 1)
            closest_indices = np.argsort(distance)[:self.k]
            y_pred = np.mean(self.Training_y[closest_indices])
            output.append(y_pred)
        return output




if __name__ == "__main__":
    training_X_file_name = "X_train_" + sys.argv[1] + ".csv"
    training_Y_file_name = "Y_train_" + sys.argv[1] + ".csv"
    testing_X_file_name = "X_test_" + sys.argv[1] + ".csv"
    testing_Y_file_name = "Y_test_" + sys.argv[1] + ".csv"

    X = Generating_Feature_Vector_Matrix(training_X_file_name)
    y = Generating_flag_value_vector(training_Y_file_name)
    X_test = Generating_Feature_Vector_Matrix(testing_X_file_name)
    y_test = Generating_flag_value_vector(testing_Y_file_name)

    Scatter_Plot(X, y, X_test, y_test, 1)
    # for i in range(1, 9):
    #     Knn_model = KNNRegression(X, y, i)
    #     y_pred = Knn_model.predict(X_test)
    #     y_diff = y_pred - y_test
    #     y_diff = y_diff.flatten()
    #     err = y_diff.dot(y_diff)
    #     print("i value: {i}, error{err}")