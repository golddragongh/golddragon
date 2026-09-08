"""
Created on 8.25.2026 3:50PM
Author: golddragon
"""

def diff(arr1, arr2):
    """
    Input: arr1, arr2 are two arrays of the same length
    Output: arr3 is the array of the discrete derivative of arr2 with respect to arr1
    """

    # Initialize an empty array to store the discrete derivative values.
    arr3 = []

    # Check if the input arrays have the same length. If not, print an error message and return None.
    if len(arr1) != len(arr2):
        print("Array sizes do not match!")
        return None

    # Calculate the discrete derivative.
    for i in range(len(arr1)):
        if i == 0:
            continue
        else:
            arr3.append((arr2[i] - arr2[i-1])/(arr1[i] - arr1[i-1]))
    return arr3