import numpy as np
import warnings 

"""
Add an extra columns of 1s to X
"""
def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)



class LinearRegression():
    
    """
    Find the optimal weight vector and create it as an 
    instance variable w (which includes the bias term), 
    using the analytical formula. 
    
    Resource: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/regression.html
    """
    def fit_analytic(self, X, y):
        X_ = pad(X)
        self.w = np.linalg.inv(X_.T@X_)@X_.T@y
    
    """
    Create instance variable w (which includes the bias term) 
    using the formula for the gradient of the loss function 
    and gradient descent
    """
    def fit_gradient(self, X, y, max_iter = 1000, alpha = 0.01):
        X_ = pad(X)
        
        # Initialize random weight vector where self.w = (w, -b) 
        self.w = np.random.rand(X_.shape[1])
        
        # Initialize new instance variable score_history
        self.score_history = []
        
        # Precompute P and q for finding the gradient 
        P = X_.T@X_
        q = X_.T@y
        
        prev_loss = np.inf # Set loss to positive infinity
        
        for i in np.arange(max_iter):
            
            # Gradient step
            gradient = (P@self.w) - q
            self.w -= alpha * gradient 
            new_loss = self.loss(X_, y)
            
            # Add value to history 
            self.score_history.append(self.score(X_, y))
            
            # Check for convergence             
            if np.allclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
            
        if not np.allclose(new_loss, prev_loss):
            warnings.warn("WARNING: Could not converge")
    
    
    """
    Returns a vector of predicted values, which are 
    the model's predictions for the values on the data X,
    which is padded.
    """
    def predict(self, X):
        return X@self.w
        
        
    """
    Return overall loss of the current weights w on data X, 
    which is padded and values y, using squared error. 
    """
    def loss(self, X, y): 
        return (self.predict(X) - y) ** 2

    """
    Determine the score of the model coefficient of determination, 
    assuming that X is padded.
    """
    def score(self, X, y):
        top_sum = np.sum(self.loss(X, y))
        y_mean = np.mean(y)
        bottom_sum = np.sum((y_mean - y) ** 2)
        return (1 - (top_sum / bottom_sum))