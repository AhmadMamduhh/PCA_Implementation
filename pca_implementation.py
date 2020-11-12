"""
Created on Thu Nov 12 13:48:30 2020

@author: Ahmed Mamdouh - 16P6020

"""
import numpy as np
import matplotlib.pyplot as plt


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
        print("You can't reduce the dimensions any further than one dimension")
        return None
    
    initialized_matrix = np.random.rand(*shape)
    
    return initialized_matrix


def apply_pca(input_matrix : np.ndarray):
    """ 
    This function applies the steps of the PCA algorithm to a numpy array of
    shape (n,d) and returns a numpy array of shape (n,k) where k < d
    
    PARAMS:
        input_matrix - The numpy array of shape (n,d) in which PCA will be applied to
        
    RETURNS:
        pc_list - A list of k principle components each of shape (n,1)
        (k = 2 for visualization)
        
    """
    
    # normalizing the input_matrix
    normalized_matrix = normalize_matrix(input_matrix)
    
    # calculating the Covariance matrix
    covariance_matrix = calculate_covariance(normalized_matrix)
    
    # calculating the eigenvalues "v" and eigenvectors "u"
    v, u = np.linalg.eig(covariance_matrix)
    
    # finding the eigenvectors with the maxium eigenvalues
    eigenvectors_list = find_max_eigenvectors(v, u,
                       input_matrix.shape[1] - 1,input_matrix.shape[0])
    
    # calculating the principle components 
    pc_list = []
    PC1 = normalized_matrix.dot(eigenvectors_list[0])
    PC1 = PC1.reshape((PC1.shape[0], 1))
    pc_list.append(PC1)
    
    if(input_matrix.shape[1] > 2):
        PC2 = normalized_matrix.dot(eigenvectors_list[1])
        PC2 = PC2.reshape((PC2.shape[0], 1))
        pc_list.append(PC2)

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


def find_max_eigenvectors(values, vectors, k, n):
    """ 
    This function takes as an input the eigenvalues, eigenvectors, the number of
    of principle components "k" and the number of examples "n" and uses them
    to return a list of k eigenvectors which correspond to the highest eigenvalues
    
    PARAMS:
        values - The numpy array of the eigenvalues
        vectors - The numpy array of the eigenvectors
        k - The number of principle components
        n - The number of data points in the datasets
        
    RETURNS:
        eigenvectors_list - The list of eigenvecotrs used for projection
        
    """
    # finding the eigenvectors with the maxium eigenvalues
    eigenvectors_list = []
    count = 0
    while count < k:
        max_index = values.argmax()
        eigenvectors_list.append(vectors[:, max_index])
        values = np.delete(values, max_index )
        count += 1
    
    return eigenvectors_list
    
    

def main():
    
    """ Main entry to the program """
    
    while True:
        
        input_matrix = initialize_random_matrix((10,3))
        if(input_matrix is None):
            break
        print(input_matrix)
        
        output_matrix = apply_pca(input_matrix)
        print(output_matrix)
        
        

        restart = input('\nWould you like to restart? Enter y or n\n')
        if restart.lower() != 'y':
            break


if __name__ == "__main__":
	main()

