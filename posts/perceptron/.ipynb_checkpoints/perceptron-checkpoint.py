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
    If p is a Perceptron object, then after p.fit(X, y) is called, 
    p should have an instance variable of weights called w. 
    (w is the vector in classifier in instructions)
    
    p should have an instance variable called p.history, 
    which is a list of the evolution of the score over the training period. 
    (see Perceptron.score(X, y) below) 
    """
    def fit(self, X, y, max_steps = 10000): #saw this defined in the project overview
        # Determine n (number of data points) from X 
        n = X.shape[0]
        
        # Modify X into X_ (which contains a column of 1s)
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        # Determine p (number of features) from X_
        p = X_.shape[1] # INCLUDING THE EXTRA COLUMN ? 
        
        # Initialize random weight vector 
        # self.w = w , -b 
        self.w = np.random.rand(p)
        
        # Define y_ which stores y as a vector of -1s and 1s instead of 0s and 1s
        y_ = (2 * y) - 1
        
        steps = 0

        self.history = []
        
        while ((self.score(X, y) != 1.0) and (steps < max_steps)):
            self.history.append(self.score(X, y))
            i = random.randint(0, n - 1)
            self.w = self.w + (((y_[i] * self.w@X_[i]) < 0) * y_[i] * X_[i])
            #print("steps:", steps)
            steps += 1
            #print("weights:", self.w)
            #print()
            
        self.history.append(self.score(X, y))
        
        if(self.score(X, y) != 1.0): 
            warnings.warn("WARNING: Could not converge")
    
    """
    Returns a vector of predicted labels (0s and 1s) 
    These are the model’s predictions for the labels on the data.
    """
    def predict(self, X): 
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1) #SHOULD WE BE MODIFYING X_
        return (X_@self.w >= 0).astype(int)
        #print("prediction:" , prediction)
        #return prediction
    
    """
    Returns the accuracy of the perceptron as a number between 0 and 1, 
    with 1 corresponding to perfect classification.
    """
    def score(self, X, y):
        #print("y:", y)
        # Creates array that indicates whether prediction was correct 
        #print("mean:", np.mean(y == self.predict(X)))
        #correct = np.sum(y == self.predict(X))
        #score = correct / y.size
        #print("score:", score)
        return np.mean(y == self.predict(X))