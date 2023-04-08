from matplotlib import pyplot as plt
import numpy as np
import warnings 

"""
Given a greyscale image (img) and the number of singular values (k) to use, 
reconstruct the image from its singular value decomposition. 
"""
def svd_reconstruct(img, k):
# MAKE THE MEMORY THING AN OPTIONAL ARGUMENT HERE


    # Get height and width from image 
    # No third value in shape since img is in greyscale
    height, width = img.shape
    
    # Get the img as a 2D matrix 
    data = np.array(img) 
    
    # Return unmodified data if k is too large
    if ((k > height) or (k > width)):
        warnings.warn("WARNING: k > height or k > width of given image")
        return data
    
    # Compute a singular value decomposition (SVD)
    U, sigma, V = np.linalg.svd(data)

    # create the D matrix in the SVD
    D = np.zeros_like(data, dtype=float) # matrix of zeros of same shape as data
    D[:min(data.shape),:min(data.shape)] = np.diag(sigma)        # singular values on the main diagonal
    
    # Approximate by using k smaller representation
    U_ = U[:,:k] # first k columns of U
    D_ = D[:k, :k] # top k singular values in D
    V_ = V[:k, :] # first k rows of V

    # Reconstruct and compute approximation of data
    data_ = U_ @ D_ @ V_
    return data_

"""
Perform an experiment that reconstructs greyscale image (img) with 
several different values of k.
"""  
def svd_experiment(img):
    rows = 3
    columns = 3
    fig, axarr = plt.subplots(rows, columns) 
    
    for i in range(rows):
        for j in range(columns):
            k = (i * 3 + j + 1) * 10
            img_title = str(k) + " components"
            axarr[i][j].imshow(svd_reconstruct(img, k), cmap="Greys")
            axarr[i][j].axis("off")
            axarr[i][j].set(title=img_title)
    
def compare_images(A, A_):

    fig, axarr = plt.subplots(1, 2, figsize = (7, 3))

    axarr[0].imshow(A, cmap = "Greys")
    axarr[0].axis("off")
    axarr[0].set(title = "original image")

    axarr[1].imshow(A_, cmap = "Greys")
    axarr[1].axis("off")
    axarr[1].set(title = "reconstructed image")