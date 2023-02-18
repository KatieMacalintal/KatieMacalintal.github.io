import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import random
import warnings 

np.random.seed(0)

class Perceptron: 
    
    """
    Determines the variable of weights (w) that linearly separates X, a dataset 
    of observations and their features, and y, the labels for each piece of data in X. 
    Stores w as an instance variable with the Perceptron object and creates the 
    instance variable history, which is a list of the evolution of the score (accuracy) 
    over the training period.
    If data is not linearly separable, a warning is raised. 
    """
    def fit(self, X, y, max_steps = 10000):
        # Determine n (number of data points) from X 
        n = X.shape[0]
        
        # Modify X into X_ (which contains a column of 1s)
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        # Determine p (number of features) from X_
        p = X_.shape[1] # INCLUDING THE EXTRA COLUMN ? 
        
        # Initialize random weight vector where self.w = (w, -b) 
        self.w = np.random.rand(p)
        
        # Define y_ which stores y as a vector of -1s and 1s instead of 0s and 1s
        y_ = (2 * y) - 1
        
        steps = 0

        self.history = []
        
        # Keep editing w until it reaches perfect classifcation or max_steps
        while ((self.score(X, y) != 1.0) and (steps < max_steps)):
            self.history.append(self.score(X, y))
            i = random.randint(0, n - 1)
            self.w = self.w + (((y_[i] * self.w@X_[i]) < 0) * y_[i] * X_[i])
            steps += 1
        self.history.append(self.score(X, y))
        
        # X is not linearly separable 
        if(self.score(X, y) != 1.0): 
            warnings.warn("WARNING: Could not converge")
    
    
    """
    Returns a vector of predicted labels (either 0s or 1s), which are the model's 
    predictions for the labels on the data X. 
    """
    def predict(self, X):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        return (X_@self.w > 0).astype(int)
    
    
    """
    Returns the accuracy of the perceptron as a number between 0 and 1, given 
    the data X and the expected labels y. 1 corresponds to perfect classification.
    """
    def score(self, X, y):
        # Calculate the average of how often model's prediction is the same as expected label 
        return np.mean(y == self.predict(X))