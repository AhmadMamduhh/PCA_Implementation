"""
Created on Thu Nov 12 13:48:30 2020

@author: Ahmed Mamdouh - 16P6020

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


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
        pc_matrix - A matrix of shape (n,k) containing all principle components
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

    # calculating the principle components and appending them to the pc_matrix
    n = input_matrix.shape[0] # number of data points
    pc_matrix = normalized_matrix.dot(eigenvectors_list[0]).reshape((n, 1)) # first principle component
    
    for vector in eigenvectors_list[1:]:
        # calculating principle component
        pc = normalized_matrix.dot(vector).reshape((n, 1))
        
        # appending pc to the principle component matrix  
        pc_matrix = np.append(pc_matrix, pc, axis= 1)
        
    return pc_matrix
    
    
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
    
    # subtracting the mean of the observations for each feature from each observation
    input_matrix -= input_matrix.mean(axis = 0) 
    
    # calculating the covariance matrix
    covariance_matrix = (np.dot(input_matrix.T, input_matrix) ) / (n - 1)
    
    # another solution: covariance_matrix = np.cov(input_matrix.T)
    
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
    # finding the eigenvectors with the maximum eigenvalues
    eigenvectors_list = []
    count = 0
    while count < k:
        max_index = values.argmax()
        eigenvectors_list.append(vectors[:, max_index])
        values = np.delete(values, max_index )
        vectors = np.delete(vectors, max_index, axis= 1)
        count += 1
    
    return eigenvectors_list
    

def display_2D_scatter_plot(dataset, title, xlabel, ylabel, labels = None):
    """ 
    This function displays the input dataset as a form of scatter plot assuming
    that the dataset is 2D only.
    
    PARAMS:
        dataset - a numpy array of shape (n,2) which has n data points and
        only 2 dimensions
        
        title - string holding the title of the figure to be displayed
        
        xlabel - string holding the label for the x axis of the plot
        
        ylabel - string holding the label for the y axis of the plot
        
        labels - the target class for each observation. This is used for visualization
        
    """
    
    plt.figure()
    plt.scatter(dataset[:,0], dataset[:,1], c = labels)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

def main():
    
    """ Main entry to the program """

    # input_matrix = initialize_random_matrix((1000,5))  # Random data points
    # if(input_matrix is None):
    #      return
    # labels = None
    
    # Loading the iris dataset
    iris_dataset = datasets.load_iris()
    input_matrix = iris_dataset.data
    labels = iris_dataset.target
    
    # Plotting first feature against the second feature
    display_2D_scatter_plot(input_matrix, "Input Dataset", "X1", "X2", labels)
        
    # Applying my implementation of PCA to the input dataset 
    pc_matrix = apply_pca(input_matrix)
    
    # Plotting the two principle components with highest variability
    display_2D_scatter_plot(pc_matrix, "Projected Dataset", "PC1", "PC2", labels)
        
           
if __name__ == "__main__":
	main()

