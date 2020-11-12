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
        output_matrix - The numpy array of shape (n,k) after PCA
        
    """
    
    # Normalizing the input_matrix
    input_matrix = normalize_matrix(input_matrix)
    
    # Calculating the Covariance matrix
    
    
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
    
    
    

def main():
    
    """ Main entry to the program """
    
    while True:
        
        input_matrix = initialize_random_matrix((20,3))
        if(input_matrix is None):
            break
        
        output_matrix = apply_pca(input_matrix)
        
        

        restart = input('\nWould you like to restart? Enter y or n\n')
        if restart.lower() != 'y':
            break


if __name__ == "__main__":
	main()

