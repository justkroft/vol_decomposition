import numpy as np
from src.vol_decomposition import multiply_array


# Create a test array
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
scalar = 2.5

# Call the C function
result = multiply_array(arr, scalar)

print("Original array:", arr)
print("Scalar:", scalar)
print("Result:", result)

# Test with 2D array
arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
result_2d = multiply_array(arr_2d, 3.0)

print("\n2D Original:", arr_2d)
print("2D Result:", result_2d)
