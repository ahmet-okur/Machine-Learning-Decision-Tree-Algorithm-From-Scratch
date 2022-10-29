#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import numpy as np
import pandas as pd


# ## Get the Data

# In[4]:


cols_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv('/Users/ahmetokur/Desktop/Datasets/iris.csv', skiprows=1, header=None, names= cols_names)
data.head(10)


# In[5]:


data.info()


# In[6]:


data.type.unique()


# In[7]:


type_col = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
data.type = [type_col[i] for i in data.type]


# In[8]:


data.type.dtype


# ## Node Class

# In[42]:


class Node():
    def __init__(self, feature_index=None, treshold=None, left=None, right=None, info_gain=None, value=None):
        """ Constructor """
        
        #for the decison node
        self.feature_index = feature_index
        self.treshold = treshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for the leaf node
        self.value = value
        


# ## Tree Class

# In[43]:


class DecisionTreeClassifier():
    def __init__(self, min_sample_split=2, max_depth=2):
        """ Constructor """
        
        # initialize the root of the tree
        self.root = None
        
        # Stoping conditions
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
    
    def buildTree(self, dataset ,curr_depth=0):
        """ recursive function to build the tree """
        
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        
        # split until stoping conditions are met
        if num_samples >= self.min_sample_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if the information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.buildTree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.buildTree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["treshold"],
                            left_subtree, right_subtree, best_split["info_gain"])
            
        # compute the leaf node
        leaf_node = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_node)
        
    def get_best_split(self, dataset, num_samples, num_features):
        """ function to find the best split """
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_tresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for treshold in possible_tresholds:
                # get the current split
                dataset_left, dataset_right = self.split(dataset, feature_index, treshold)
                # chech if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["treshold"] = treshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
            
        # return best split
        return best_split
    
    def split(elf, dataset, feature_index, treshold):
        """ function to split the data """
        
        dataset_left = np.array([row for row in dataset if row[feature_index] <= treshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > treshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        """ function to compute the information gain """
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if(mode == "gini"):
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child)) + (weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child)) + (weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        """ function to compute the entropy """
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        """ function to compute the gini index """
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, y):
        """ function to compute the leaf node """
        
        y = list(y)
        return max(y, key=y.count)
    
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        
        else:
            print("X_"+ str(tree.feature_index), "<=", tree.treshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%right:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        """ function to train the tree """
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataset)
    
    def predict(self, X):
        """ function to predict new dataset """
        
        predictions = [self.make_prediction(X, self.root) for X in X]
        return predictions
    
    def make_prediction(self, x, tree):
        """ function to predict a single data point """
        
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.treshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        


# ## Train-Test-Split

# In[52]:


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# ## Fit the Model

# In[53]:


classifier = DecisionTreeClassifier(min_sample_split=3, max_depth=3)
classifier.fit(x_train, y_train)
classifier.print_tree()


# ## Test the Model

# In[54]:


y_pred = classifier.predict(x_test)


# In[55]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:




