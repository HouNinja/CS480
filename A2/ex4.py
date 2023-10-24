import numpy as np

class DecisionTree:
    #You will likely need to add more arguments to the constructor
    def __init__(self):
        #Implement me!
        return

    def build(self, X, y):
        #Implement me!
        return
    
    def predict(self, X):
        #Implement me!
        return 

#Load data
X_train = np.loadtxt('data/X_train_D.csv', delimiter=",")
y_train = np.loadtxt('data/y_train_D.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test_D.csv', delimiter=",")
y_test = np.loadtxt('data/y_test_D.csv', delimiter=",").astype(int)

