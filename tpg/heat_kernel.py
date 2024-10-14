import numpy as np

def taylor_expansion(x, terms=5):
    """
    Compute the Taylor series approximation of exp(x) up to a specified number of terms.
    
    需要注意，泰勒展开式的前提是在展开处（即0）二者等价无穷小

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


def heat_kernel(vectors, lambda_param=1.0, taylor_terms=5):
    """
    计算输入向量两两间的相似度。
    
    Parameters
    ----------
    vectors: np.array
        The input list of vectors, 2-D matrix.
    lambda_param : float
        The time parameter λ for the heat kernel function.
    taylor_terms : int
        The number of terms to use in the Taylor series expansion.
    
    Returns
    -------
    similarity : np.array
        The similarity between assets, within 0 and 1.
    """
    assert(len(vectors) >= 2), "Heat Kernel takes at least 2 vectors as input!"
    assert(len(vectors.shape) == 2), "Heat Kernel only processes 2-D matrix inputs!"

    # The first value of shape indicates the total number of vectors.
    num_v = vectors.shape[0]
    similarities = np.zeros((num_v, num_v))
    for i in range(0, num_v):
        for j in range(i, num_v):
            if j == i: continue
            # Calculate the squared Euclidean distance between the two vectors
            distance_sq = np.sum((vectors[i] - vectors[j])**2)

            # Limit distance to prevent numerical issues with large values
            distance_sq = np.clip(distance_sq, 0, 1e6)  # Cap the distance for stability
            
            # Calculate the argument for the exponential function
            exp_argument = -distance_sq / lambda_param

            # Ensure exp_argument is within a reasonable range to avoid overflow
            if exp_argument < -10:
                # If the argument is too negative, return a small similarity directly (since e^(-large) is ~0)
                similarity = 0
            # 仅当x距离0较近时才使用泰勒展开，否则不能进行拟合
            elif exp_argument < -2:
                similarity = np.exp(exp_argument)
            else:
                similarity = taylor_expansion(exp_argument, terms=taylor_terms)

            # Ensure the similarity is between 0 and 1 (it can grow due to Taylor expansion)
            similarity = np.clip(similarity, 0, 1)
            
            similarities[i][j] = similarity
            similarities[j][i] = similarity
    
    return similarities


# Example usage
if __name__ == '__main__':
    # Create two random 128-dimensional vectors (example input)
    vector_i = np.random.rand(128)
    vector_j = np.random.rand(128)
    
    # Compute similarity using Taylor expansion of heat kernel
    lambda_param = 1.0  # Set the time parameter λ
    taylor_terms = 7  # Number of terms in the Taylor series
    similarity = heat_kernel(vector_i, vector_j, lambda_param, taylor_terms)
    
    print(f"Taylor-approximated similarity between vectors: {similarity:.4f}")
