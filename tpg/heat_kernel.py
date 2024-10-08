import numpy as np

def taylor_expansion(x, terms=5):
    """
    Compute the Taylor series approximation of exp(x) up to a specified number of terms.
    
    需要注意，泰勒展开式的前提是二者等价无穷小

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

    # Limit distance to prevent numerical issues with large values
    distance_sq = np.clip(distance_sq, 0, 1e6)  # Cap the distance for stability
    
    # Calculate the argument for the exponential function
    exp_argument = -distance_sq / lambda_param

    # Ensure exp_argument is within a reasonable range to avoid overflow
    if exp_argument < -10:
        # If the argument is too negative, return a small similarity directly (since e^(-large) is ~0)
        return 0.0
    
    # Use Taylor series to approximate the exponential function
    # TODO 泰勒展开式目前不太对
    # similarity = taylor_expansion(exp_argument, terms=taylor_terms)
    similarity = np.exp(exp_argument)

    # Ensure the similarity is between 0 and 1 (it can grow due to Taylor expansion)
    similarity = np.clip(similarity, 0, 1)
    
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
