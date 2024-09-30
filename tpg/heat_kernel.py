import numpy as np

def taylor_expansion(x, terms=5):
    """
    Compute the Taylor series approximation of exp(x) up to a specified number of terms.
    
    Parameters
    ----------
    x : float
        The input to the exponential function.
    terms : int
        The number of terms to use in the Taylor series expansion.
    
    Returns
    -------
    approx_exp : float
        The approximation of exp(x) using Taylor series.
    """
    approx_exp = 0
    for n in range(terms):
        approx_exp += (x**n) / np.math.factorial(n)
    return approx_exp


def heat_kernel(e_i, e_j, lambda_param=1.0, taylor_terms=5):
    """
    Compute the similarity between two vectors e_i and e_j using the heat kernel
    approximation via Taylor series.
    
    Parameters
    ----------
    e_i : ndarray (128,)
        Embedding vector or feature vector of asset i.
    e_j : ndarray (128,)
        Embedding vector or feature vector of asset j.
    lambda_param : float
        The time parameter λ for the heat kernel function.
    taylor_terms : int
        The number of terms to use in the Taylor series expansion.
    
    Returns
    -------
    similarity : float
        The similarity between asset i and asset j, between 0 and 1.
    """
    # Calculate the squared Euclidean distance between the two vectors
    distance_sq = np.sum((e_i - e_j)**2)
    
    # Calculate the argument for the exponential function
    exp_argument = -distance_sq / lambda_param
    
    # Use Taylor series to approximate the exponential function
    similarity = taylor_expansion(exp_argument, terms=taylor_terms)
    
    return similarity


# Example usage
if __name__ == '__main__':
    # Create two random 128-dimensional vectors (example input)
    vector_i = np.random.rand(128)
    vector_j = np.random.rand(128)
    
    # Compute similarity using Taylor expansion of heat kernel
    lambda_param = 1.0  # Set the time parameter λ
    taylor_terms = 5  # Number of terms in the Taylor series
    similarity = heat_kernel(vector_i, vector_j, lambda_param, taylor_terms)
    
    print(f"Taylor-approximated similarity between vectors: {similarity:.4f}")
