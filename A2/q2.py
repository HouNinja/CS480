import numpy as np 
import pandas as pd
import sys


def Compute_error(X, y, w, b, C, epsilon):
    Loss = np.linalg.norm(w, ord = 2) / 2
    error = 0
    for index, row in X.iterrows():
        error += C * max(abs(y[index] - (np.dot(w, row) + b)) - epsilon, 0)
    
    Loss += error
    return Loss, error
    

def SVR_Gradient_Descent(X, y, num_iterations, C, epsilon):
    row, column = X.shape
    print(row, column)
    w_0 = np.zeros(column)
    b_0 = 0
    # vector_one = np.ones((row, 1))
    # direction = np.zeros((column, 1))
    step_size = 0.4
    for _ in range(num_iterations):
        for index, row in X.iterrows():
            # stepsize
            prev_w_0 = w_0
            if y[index] - np.dot(row, prev_w_0) - b_0 >= epsilon:
                direction = -1 * C * row
                w_0 = w_0 - step_size * direction
                b_0 = b_0 - step_size * (-1) * C
            elif y[index] - np.dot(row, prev_w_0) - b_0 <= -1 * epsilon:
                direction = C * row
                w_0 = w_0 - step_size * direction
                b_0 = b_0 - step_size * C
            w_0 = w_0 / (1 + step_size)

    return w_0, b_0

if __name__ == "__main__":
    Training_file_dataset_name = "data/X_train_" + sys.argv[1] + ".csv"
    Training_file_flag_name = "data/Y_train_" + sys.argv[1] + ".csv"

    data = pd.read_csv(Training_file_dataset_name, header = None)
    training_y = pd.read_csv(Training_file_flag_name, header = None)
    training_y = training_y.iloc[:, 0]

    w, b = SVR_Gradient_Descent(data, training_y, 100, 1, 0.5)

    Training_Loss, Training_Error = Compute_error(data, training_y, w, b, 1, 0.5)
    print(f"the Training loss is {Training_Loss}")
    print(f"the Training error is {Training_Error}")

    Test_file_dataset_name = "data/X_test_" + sys.argv[1] + ".csv"
    Test_file_flag_name = "data/Y_test_" + sys.argv[1] + ".csv"

    test_data = pd.read_csv(Test_file_dataset_name, header = None)
    test_y = pd.read_csv(Test_file_flag_name, header = None)
    test_y = test_y.iloc[:, 0]

    _, test_error = Compute_error(test_data, test_y, w, b, 1, 0.5)
    print(f"the Test error is {test_error}")

    print(w, b)