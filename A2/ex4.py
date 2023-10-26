import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, label, data_points, labels):
        self.label = label
        self.data_points = data_points
        self.labels = labels
        self.dimension = -1
        self.bound = 0
        self.left = None
        self.right = None



class DecisionTree:
    #You will likely need to add more arguments to the constructor

    def __init__(self, hyperparameter):
        #Implement me!
        self.root = None
        self.hyperparameter = hyperparameter
        return

    def Compute_Loss(self, Data_Points, Data_labels, loss_function):
        num_rows, num_columns = Data_Points.shape
        if num_rows == 0:
            return 0, 0
        
        count_ones = 0
        for flag in Data_labels:
            if flag == 1:
                count_ones += 1

        loss, predicted_label = 0, 0
        ratio_one = count_ones / num_rows

        if num_rows - count_ones > count_ones:
            predicted_label = 0
        else:
            predicted_label = 1

        if loss_function == "Misclassification_Loss":
            if num_rows - count_ones > count_ones:
                loss = ratio_one
            else:
                loss = 1 - ratio_one

        elif loss_function == "Entropy":
            if ratio_one == 1:
                return 0, 1
            elif ratio_one == 0:
                return 0, 0
            loss = -1 * ( ratio_one * math.log2(ratio_one) + (1 - ratio_one) * math.log2(1 - ratio_one))
        elif loss_function == "Gini_Index":
            loss = ratio_one * (1 - ratio_one) * 2

        return loss, predicted_label


    def Split(self, node, loss_function):
        num_rows, num_columns = node.data_points.shape
        Data_L, Data_R, y_L, y_R, dimension, bound = pd.DataFrame(), pd.DataFrame(), None, None, -1, 0
        Predicted_label_L, Predicted_label_R = 0, 0

        min_loss = float("inf")
        for i in range(num_columns):
            for index in range(num_rows):
                Temp_Data_L = []
                Temp_Data_R = []
                for index_1 in range(num_rows):
                    if node.data_points[index_1][i] <= node.data_points[index][i]:
                        Temp_Data_L.append(index_1)
                    else:
                        Temp_Data_R.append(index_1)
                size_L = len(Temp_Data_L)
                size_R = len(Temp_Data_R)
                Temp_y_L = [node.labels[i] for i in Temp_Data_L]
                Temp_y_R = [node.labels[i] for i in Temp_Data_R]
                Temp_Data_L = node.data_points[Temp_Data_L, :]
                Temp_Data_R = node.data_points[Temp_Data_R, :]
                Loss_L, Temp_predicted_label_L = self.Compute_Loss(Temp_Data_L, Temp_y_L, loss_function)
                # print(f"length of R is {size_R}")
                Loss_R, Temp_predicted_label_R = self.Compute_Loss(Temp_Data_R, Temp_y_R, loss_function)
                if min_loss > size_L * Loss_L + size_R * Loss_R:
                    Data_L = Temp_Data_L
                    Data_R = Temp_Data_R
                    y_L = Temp_y_L
                    y_R = Temp_y_R
                    dimension = i
                    bound = node.data_points[index][i]
                    Predicted_label_L = Temp_predicted_label_L
                    Predicted_label_R = Temp_predicted_label_R
                    min_loss = size_L * Loss_L + size_R * Loss_R
        
        if Data_L.size != 0 and Data_R.size != 0:
            node.left = TreeNode(Predicted_label_L, Data_L, y_L)
            node.right = TreeNode(Predicted_label_R, Data_R, y_R)
        node.dimension = dimension
        node.bound = bound
        return

    def recursively_Split(self, node, loss_function, level):
        if node == None or level > self.hyperparameter:
            return
        self.Split(node, loss_function)
        self.recursively_Split(node.left, loss_function, level + 1)
        self.recursively_Split(node.right, loss_function, level + 1)
        return


    def build(self, X, y, loss_function):
        #Implement me!
        self.root = TreeNode(0, X, y)
        output = self.recursively_Split(self.root, loss_function, 0)
        return
    
    def predict(self, X):
        #Implement me!
        output = []
        for index, row in enumerate(X):
            cur_node = self.root
            while cur_node.left != None:
                # print(cur_node.dimension)
                # print(cur_node.bound)
                # print(len(cur_node.data_points))
                if row[cur_node.dimension] <= cur_node.bound:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right
            output.append(cur_node.label)

        return output

def Accuracy_Test(predicted_y, y):
    tol = len(predicted_y)
    correct_prediction = 0
    for i in range(tol):
        if predicted_y[i] == y[i]:
            correct_prediction += 1
    return correct_prediction / tol

def plot_the_graph(X, y, X_test, y_test, loss_function):
    X_axis = []
    Y_axis = []
    Y2_axis = []

    for i in range(1, 51):
        X_axis.append(i)
        Tree = DecisionTree(i)
        Tree.build(X, y, loss_function)
        predicted_y = Tree.predict(X_test)
        predicted_on_training_y = Tree.predict(X)
        Y_axis.append(Accuracy_Test(predicted_on_training_y, y_train))
        Y2_axis.append(Accuracy_Test(predicted_y, y_test))

    plt.plot(X_axis, Y_axis, marker='o', linestyle='-')
    plt.plot(X_axis, Y2_axis, marker='o', linestyle='-')
    plt.xlabel('Hyperparameter')
    plt.ylabel('Accuracy rate')
    plt.title('Connected Scatter Plot')
    plt.show()


        
#Load data
if __name__ == "__main__":
    X_train = np.loadtxt('data/X_train_D.csv', delimiter=",")
    y_train = np.loadtxt('data/y_train_D.csv', delimiter=",").astype(int)
    X_test = np.loadtxt('data/X_test_D.csv', delimiter=",")
    y_test = np.loadtxt('data/y_test_D.csv', delimiter=",").astype(int)

    plot_the_graph(X_train, y_train, X_test, y_test, "Gini_Index")
    

