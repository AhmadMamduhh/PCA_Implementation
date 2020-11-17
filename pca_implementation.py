"""
Created on Thu Nov 12 13:48:30 2020

@author: Ahmed Mamdouh - 16P6020

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def initialize_random_matrix(shape : tuple):
    """ 
    This function creates a numpy array with a specified shape and fills
    it with random numbers
    
    PARAMS:
        shape - A tuple which indicates the shape of the numpy array to be returned
        
    RETURNS:
        initialized_matrix - A numpy array filled with random numbers
        
    """
    if(shape[1] == 1):
        print("Please choose a dimension for the data higher than 1!")
        return None
    
    initialized_matrix = np.random.rand(*shape)
    
    return initialized_matrix


def apply_pca(input_matrix : np.ndarray):
    """ 
    This function applies the steps of the PCA algorithm to a numpy array of
    shape (n,d) and returns a numpy array of shape (n,k)
    
    PARAMS:
        input_matrix - The numpy array of shape (n,d) in which PCA will be applied to
        
    RETURNS:
        pc_list - A list of k principle components each of shape (n,1)
        (k == d for this script)
        
    """
    
    # normalizing the input_matrix
    normalized_matrix = normalize_matrix(input_matrix)
    
    # calculating the Covariance matrix
    covariance_matrix = calculate_covariance(normalized_matrix)
    
    # calculating the eigenvalues "v" and eigenvectors "u"
    v, u = np.linalg.eig(covariance_matrix)
    
    # Sorting the eigenvectors such that the vectors with higher eigenvalues are placed first
    eigenvectors_list = sort_eigenvectors(v, u, k = input_matrix.shape[1]) # values, vectors, number of dimensions
    
    # calculating the principle components and appending them to the pc_list
    pc_list = []
    
    for vector in eigenvectors_list:
        PC = normalized_matrix.dot(vector)
        PC = PC.reshape((PC.shape[0], 1))
        pc_list.append(PC)


    return pc_list
    
    
    
def normalize_matrix(input_matrix : np.ndarray):
    """ 
    This function takes as an input a numpy array and subtracts the mean of
    its values from each value and divides by the standard deviation.
    
    PARAMS:
        input_matrix - The numpy array to be normalized
        
    RETURNS:
        normalized_matrix - The numpy array after normalization
        
    """
    normalized_matrix = input_matrix - np.mean(input_matrix)
    normalized_matrix /= np.std(normalized_matrix)
    return normalized_matrix


def calculate_covariance(input_matrix : np.ndarray):
    """ 
    This function takes as an input a numpy array and calculates its covariance
    matrix
    
    PARAMS:
        input_matrix - The numpy array of which we're going to calculate the
        covariance matrix 
        
    RETURNS:
        covariance_matrix - The covariance matrix of the input_matrix
        
    """
    # calculating the number of data points "n" 
    n = input_matrix.shape[0]
    
    # calculating the covariance matrix
    covariance_matrix = (np.dot(input_matrix.T, input_matrix)) / n
    
    return covariance_matrix


def sort_eigenvectors(values, vectors, k = 2):
    """ 
    This function takes as an input the eigenvalues, eigenvectors and the number of
    of principle components "k" and uses themto return a sorted list of k eigenvectors
    which correspond to the highest eigenvalues
    
    PARAMS:
        values - The numpy array of the eigenvalues
        vectors - The numpy array of the eigenvectors
        k - The number of principle components
        
    RETURNS:
        eigenvectors_list - The list of eigenvectors used for projection
        
    """
    # finding the eigenvectors with the maxium eigenvalues
    eigenvectors_list = []
    count = 0
    while count < k:
        max_index = values.argmax()
        eigenvectors_list.append(vectors[:, max_index])
        values = np.delete(values, max_index )
        vectors = np.delete(vectors, max_index, axis= 1)
        count += 1
    
    return eigenvectors_list
    

def apply_pca_scikit(input_matrix):
    """ 
    This function applies the PCA algorithm to a numpy array using the SciKit library.
    I used this function mainly to check my results.
    
    PARAMS:
        input_matrix - The numpy array of shape (n,d) in which PCA will be applied to
        
    RETURNS:
        output_matrix - The numpy array of shape (n,k) after applying PCA 
        (k == d for this script)
        
    """
    # preprocessing step: normalize the data
    normalized_matrix = normalize_matrix(input_matrix)
    
    # PCA algorithm steps
    pca = PCA(n_components= 2)
    pca.fit(normalized_matrix)
    output_matrix = pca.transform(normalized_matrix)
    
    return output_matrix


def display_2D_scatter_plot(dataset, title, xlabel, ylabel):
    """ 
    This function displays the input dataset as a form of scatter plot assuming
    that the dataset is 2D only.
    
    PARAMS:
        dataset - a numpy array of shape (n,2) which has n data points and
        only 2 dimensions
        
        title - string holding the title of the figure to be displayed
        
        xlabel - string holding the label for the x axis of the plot
        
        ylabel - string holding the label for the y axis of the plot
        
    """
    plt.figure()
    
    if(type(dataset) == list): # Because I store the principle components in a list data structure
        plt.scatter(dataset[0], dataset[1])
    else:
        plt.scatter(dataset[:,0], dataset[:,1])
      
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

def main():
    
    """ Main entry to the program """

    input_matrix = initialize_random_matrix((1000,5))
    if(input_matrix is None):
        return
        
    display_2D_scatter_plot(input_matrix, "Input Dataset", "X1", "X2")
        
    pc_matrix = apply_pca(input_matrix)
    display_2D_scatter_plot(pc_matrix, "Projected Dataset", "PC1", "PC2")
        
    pc_matrix_scikit = apply_pca_scikit(input_matrix)
    display_2D_scatter_plot(pc_matrix_scikit, "Projected Dataset Using SciKit", "PC1", "PC2")
        
        

if __name__ == "__main__":
	main()

