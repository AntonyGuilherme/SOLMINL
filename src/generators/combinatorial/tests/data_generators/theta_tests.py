import numpy as np
import random
import os

def _generate_easy_thetas(n, m, normal_range, small_range):
    """
    Generates a Thetas matrix where the first theta is way smaller.
    """
    num_cols = n - 1 
    thetas_matrix = np.zeros((m, num_cols))
    for i in range(m):
        for j in range(num_cols):
            if i == 0:
                thetas_matrix[i, j] = random.uniform(*small_range)
            else:
                thetas_matrix[i, j] = random.uniform(*normal_range)
    return thetas_matrix

def _generate_difficult_thetas(n, m, normal_range, small_range):
    """
    Generates a Thetas matrix where one non-first theta is way smaller.
    """
    num_cols = n - 1
    thetas_matrix = np.zeros((m, num_cols))
    small_row_idx = random.randint(2, num_cols)
    for i in range(m):
        for j in range(num_cols):
            if i == small_row_idx:
                thetas_matrix[i, j] = random.uniform(*small_range)
            else:
                thetas_matrix[i, j] = random.uniform(*normal_range)
    return thetas_matrix


def write_theta(thetas_matrix, output_filepath):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Write to file
    with open(output_filepath, 'w') as f:
        for row in thetas_matrix:
            f.write(' '.join(f"{val:.6f}" for val in row) + '\n')


def generate_difficult_thetas_file(permutation_size, number_of_permuations, output_filepath,
                         normal_range=(4, 8), small_range=(0.5, 1)):
    """
    Generates a Thetas.txt file based on specified instance type.

    Args:
        permutation_size (int): Permutation size (determines number of columns: n-1).
        number_of_permuations (int): Number of Generalized Mallows models (determines number of rows).
        output_filepath (str): Path to the output Thetas.txt file.
        normal_range (tuple): (min, max) for "normal" theta values.
        small_range (tuple): (min, max) for "way smaller" theta values.
    """

    thetas_matrix = _generate_difficult_thetas(permutation_size, number_of_permuations, normal_range, small_range)
    
    write_theta(thetas_matrix, output_filepath)

    print(f"Generated difficult Thetas.txt file at: {output_filepath}")


def generate_easy_thetas_file(permutation_size, number_of_permuations, output_filepath,
                         normal_range=(4, 8), small_range=(0.5, 1)):
    """
    Generates a Thetas.txt file based on specified instance type.

    Args:
        permutation_size (int): Permutation size (determines number of columns: n-1).
        number_of_permuations (int): Number of Generalized Mallows models (determines number of rows).
        output_filepath (str): Path to the output Thetas.txt file.
        normal_range (tuple): (min, max) for "normal" theta values.
        small_range (tuple): (min, max) for "way smaller" theta values.
    """

    thetas_matrix = _generate_difficult_thetas(permutation_size, number_of_permuations, normal_range, small_range)
    
    write_theta(thetas_matrix, output_filepath)

    print(f"Generated easy Thetas.txt file at: {output_filepath}")
