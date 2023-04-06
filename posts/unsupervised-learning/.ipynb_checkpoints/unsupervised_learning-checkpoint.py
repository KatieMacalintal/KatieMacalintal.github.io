from matplotlib import pyplot as plt
import numpy as np

"""
Write a function called svd_reconstruct that 
reconstructs an image from its singular value decomposition. 

Your function should have two arguments: the image to reconstruct, and the number k of singular values to use.
"""
def svd_reconstruct(img, k):
    height, width = img.shape
    
    data = np.array(img) # SOME OF THE VALUES ARE NEGATIVE
    
    # Specified a too large k 
    if ((k > height) or (k > width)):
        # SHOULD PROBABLY DO SOMETHING ELSE HERE AS WELL 
        return data
    
    U, sigma, V = np.linalg.svd(data)
    # create the D matrix in the SVD
    D = np.zeros_like(data, dtype=float) # matrix of zeros of same shape as A
    D[:min(data.shape),:min(data.shape)] = np.diag(sigma)        # singular values on the main diagonal
    
    
    U_ = U[:,:k]
    D_ = D[:k, :k]
    V_ = V[:k, :]
    data_ = U_ @ D_ @ V_
    return data_
    
def experiment(img):
    #img is greyscale
    fig, axarr = plt.subplots(3, 3, figsize = (7, 3))
    
    
    axarr[0][0].imshow(svd_reconstruct(img, 5), cmap = "Greys")
    axarr[0][0].axis("off")
    axarr[0][0].set(title = "fjdskl")
    
    axarr[0][1].imshow(svd_reconstruct(img, 10), cmap = "Greys")
    axarr[0][1].axis("off")
    axarr[0][1].set(title = "fjdskl")
    
    axarr[0][2].imshow(svd_reconstruct(img, 15), cmap = "Greys")
    axarr[0][2].axis("off")
    axarr[0][2].set(title = "fjdskl")
    
    axarr[1][0].imshow(svd_reconstruct(img, 20), cmap = "Greys")
    axarr[1][0].axis("off")
    axarr[1][0].set(title = "fjdskl")
    
    axarr[1][1].imshow(svd_reconstruct(img, 25), cmap = "Greys")
    axarr[1][1].axis("off")
    axarr[1][1].set(title = "fjdskl")
    
    axarr[1][2].imshow(svd_reconstruct(img, 30), cmap = "Greys")
    axarr[1][2].axis("off")
    axarr[1][2].set(title = "fjdskl")
    
    axarr[2][0].imshow(svd_reconstruct(img, 35), cmap = "Greys")
    axarr[2][0].axis("off")
    axarr[2][0].set(title = "fjdskl")
    
    axarr[2][1].imshow(svd_reconstruct(img, 40), cmap = "Greys")
    axarr[2][1].axis("off")
    axarr[2][1].set(title = "fjdskl")
    
    axarr[2][2].imshow(svd_reconstruct(img, 45), cmap = "Greys")
    axarr[2][2].axis("off")
    axarr[2][2].set(title = "fjdskl")
    

# DO THE FIRST OPTIONAL EXTRA 
    