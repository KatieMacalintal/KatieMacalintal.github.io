import numpy as np
import warnings 

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)

class LinearRegression():
    
    """
    It’s fine for you to either define separate methods like 
    fit_analytic and fit_gradient for these methods. 
    It’s also fine to define a single fit method with a 
    method argument to determine which algorithm is used.
    """
    def fit_analytic(self, X, y):
        X_ = pad(X)
        self.w = np.linalg.inv(X_.T@X_)@X_.T@y
        # added (array)[:,np.newaxis]
    
    def fit_gradient(self, X, y, max_iter = 1000, alpha = 0.01):
        X_ = pad(X)
        
        # Initialize random weight vector where self.w = (w, -b) 
        self.w = np.random.rand(X_.shape[1])
        # had self.w = np.random.rand(X_.shape[1], 1)
        
        # Initialize new instance variables 
        self.score_history = []
        
        # Precompute P and q for finding the gradient 
        P = X_.T@X_
        #print(f"{np.shape(P)}\n{P=}\n")
        q = X_.T@y
        #print(f"{np.shape(q)}\n{q=}\n")
        
        prev_loss = np.inf
        
        for i in np.arange(max_iter):
            
            # Gradient step
            gradient = (P@self.w) - q
            print(f"{np.shape(gradient)}\n{gradient=}\n")
            print(f"{np.shape(self.w)}\n{self.w=}\n")
            self.w -= alpha * gradient 
            new_loss = self.loss(X_, y)
            
            # Add value to history 
            self.score_history.append(self.score(X_, y))
            
            # Check for convergence 
            # Note: Could also check for convergence using the change in w 
            
            if np.allclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
            
            # if np.allclose(gradient, np.zeros(len(gradient))):
            #     #print(f"{self.w=}\n")
            #     break 
        
        # if not np.allclose(gradient, np.zeros(len(gradient))):
        #     warnings.warn("WARNING: Could not converge")
    
    
    def predict(self, X):
        print(f"{np.shape(X)=}")
        print(f"{np.shape(self.w)=}")
        return X@self.w
        
        
    def loss(self, X, y): 
        # CHECK THIS FUNCTION
        return (self.predict(X) - y) ** 2

    """
    The coefficient of determination is always no larger than 1, 
    with a higher value indicating better predictive performance. 
    It can be arbitrarily negative for very bad models.
    Note that the numerator in the fraction is just, so 
    making the loss small makes the coefficient of determination large.
    """
    def score(self, X, y):
        #ASSUMES X IS PADDED 
        y_hat = self.predict(X)
        top_sum = np.sum((y_hat - y) ** 2)
        y_mean = np.mean(y)
        bottom_sum = np.sum((y_mean - y) ** 2)
        return (1 - (top_sum / bottom_sum))