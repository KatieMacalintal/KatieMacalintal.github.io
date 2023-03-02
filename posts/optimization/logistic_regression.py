import numpy as np
from scipy.optimize import minimize
import warnings 
np.seterr(all='ignore') 

class LogisticRegression():
    
    """
    Determines the variable of weights w (including the bias term b), such that 
    it separates data X into their respective labels using the gradient 
    descent framework and logistic loss. 
    
    Creates instance variables of weights w, 
    loss_history (list of the evolution of the loss), and 
    score_history (list of the evolution of the score, see score(X, y). 
    """
    def fit(self, X, y, alpha = 0.001, max_epochs = 10000): 
        # Add constant feature to feature matrix 
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        # Determine p (number of features) from X_
        p = X_.shape[1]
        
        # Initialize random weight vector where self.w = (w, -b) 
        self.w = np.random.rand(p)

        # Initialize new instance variables 
        self.score_history = []
        self.loss_history = []
        
        prev_loss = np.inf # Set loss to positive infinity
        
        # Compute gradient descent
        #     Reference: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
        for i in np.arange(max_epochs):
            
            gradient = self.gradient(X_, y)
            
            # Gradient step 
            self.w -= alpha * self.gradient(X_, y)                
            new_loss = self.loss(X_, y) # compute loss
            
            # Add values to history 
            self.score_history.append(self.score(X_, y))
            self.loss_history.append(self.loss(X_, y)) 
            
            # Check if change in loss is close, terminate
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
            
        if not (np.isclose(new_loss, prev_loss)):
            warnings.warn("WARNING: Could not converge")
        
    """
    Determines the variable of weights w (including the bias term b), such that 
    it separates data X into their respective labels using stochastic gradient 
    descent and logistic loss. 
    
    Creates instance variables of weights w, 
    loss_history (list of the evolution of the loss), and 
    score_history (list of the evolution of the score, see score(X, y). 
    """
    def fit_stochastic(self, X, y, alpha = 0.001, max_epochs = 100, batch_size = 10, momentum = False):
        
        # Add constant feature to feature matrix 
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        # Determine p (number of features) from X_
        p = X_.shape[1]

        # Number of data points 
        n = X_.shape[0]
        
        # Initialize random weight vector where self.w = (w, -b) 
        self.w = np.random.rand(p)
        prev_w = np.array(self.w)
        
        # Initialize new instance variables 
        self.score_history = []
        self.loss_history = []

        # Set momentum appropriately 
        if (momentum == True):
            beta = 0.8
        else:
            beta = 0
        
        prev_loss = np.inf # Set loss to positive infinity

        # Compute gradient 
        #     Provided in description: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
        for j in np.arange(max_epochs):

            order = np.arange(n)
            np.random.shuffle(order)
            
            # Compute gradient on batches 
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:] #extract features 
                y_batch = y[batch] #extract targets 
                gradient = self.gradient(x_batch, y_batch) 
                
                # Perform gradient step
                self.w = self.w - (alpha * gradient) + (beta * (self.w - prev_w))
                prev_w = np.array(self.w) #what should the previous w be initalized to

            new_loss = self.loss(X_, y) # compute loss  
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
            
            self.score_history.append(self.score(X_, y))
            self.loss_history.append(self.loss(X_, y))
        
        if not (np.isclose(new_loss, prev_loss)):
            warnings.warn("WARNING: Could not converge")
    
    """
    Returns a vector of predicted labels, which are 
    the model's predictions for the labels on the data X. 
    
    Reference: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
    """
    def predict(self, X): 
        return X@self.w
    
    """
    Returns the accuracy of the predictions as a number between 0 and 1, 
    given the data X and the expected labels y, with 1 corresponding 
    to perfect classification.
    """
    def score(self, X, y): 
        # Need to convert prediction to label 1 or 0
        # since y are labels of 1s and 0s
        return np.mean(y == 1 *(self.predict(X) > 0))
    
    """ 
    Return overall loss (empirical risk) of the current weights w on data X and labels y,
    using logistic loss. 
    
    Reference: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
    """
    def loss(self, X, y): 
        # Compute predictions 
        y_ = self.predict(X) 
        #Compute average loss per observation
        loss = (-y * np.log(self.sigmoid(y_))) - ((1 - y) * np.log(1 - self.sigmoid(y_)))
        return loss.mean() 
    
    """
    Logistic signmoid function for logistic loss. 
    
    Reference: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    """
    Calculate the gradient of our loss function, which uses logistic loss 
    """
    def gradient(self, X, y):
        # Equation Reference: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html 
        y_ = self.predict(X)
        return np.mean(((self.sigmoid(y_) - y)[:,np.newaxis]) * X, axis = 0)