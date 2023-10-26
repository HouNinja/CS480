import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.discrete.discrete_model import Logit as Logit
from sklearn.svm import SVC
import sys

def Compute_Misclassified_Point(Model, X, y):
    weights = Model.coef_[0]
    output = 0
    cache = []
    for index, row in X.iterrows():
        flag = y[index]
        if y[index] == 0:
            flag = -1
        inner_product = (np.dot(weights, row) + Model.intercept_[0]) * flag
        cache.append(inner_product)
        if inner_product < 0.9999999:
            output += 1
    return output


if __name__ == "__main__":
    Training_file_dataset_name = "data/X_train_" + sys.argv[1] + ".csv"
    Training_file_flag_name = "data/Y_train_" + sys.argv[1] + ".csv"

    data = pd.read_csv(Training_file_dataset_name, header = None)
    training_y = pd.read_csv(Training_file_flag_name, header = None)
    data_with_intercept = sm.tools.add_constant(data)
    model = Logit(training_y, data_with_intercept)
    result = model.fit()

    clf_soft = SVC(kernel = 'linear', C = 1.0)
    training_y = training_y.iloc[:, 0]
    clf_soft.fit(data, training_y)
    print(len(clf_soft.support_))

    clf_hard = SVC(kernel = 'linear', C = float('inf'))
    # clf_hard.fit(data,training_y)

    num = Compute_Misclassified_Point(clf_soft, data, training_y)
    print(num)

