import random
import os
import kendall_distance_tests as distance
import theta_tests as theta
import numpy as np

def generate_permutations_file(num_permutations, permutation_size, output_filepath: str, typ: str = ""):
    """
    Generates a file with random permutations.

    Args:
        num_permutations (int): The number of permutations (rows) to generate.
        permutation_size (int): The size of each permutation (number of elements per row) 
                                and also defines the range of elements (1 to m).
        output_filepath (str): Path to the output file.
        typ (str): 'easy' || 'difficult' || 'similar':
    """
    if num_permutations <= 0:
        raise ValueError("Number of permutations must be greater than 0.")
    if permutation_size <= 0:
        raise ValueError("Permutation size must be greater than 0.")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    permutations = []

    with open(output_filepath, 'w') as f:
        for _ in range(num_permutations):
            # Create a list of elements from 1 to permutation_size
            elements = list(range(1, permutation_size + 1))
            # Shuffle the list to get a random permutation
            random.shuffle(elements)

            permutations.append(elements)

            # Write the permutation to the file, space-separated
            f.write(' '.join(map(str, elements)) + '\n')

    output_distance = output_filepath.replace(".txt", "_distance.txt")
    distance.generate_kendall_distances_file(permutations, output_distance)


    output_theta = output_filepath.replace(".txt", "_theta.txt")
    if typ == "easy":
        theta.generate_easy_thetas_file(permutation_size, num_permutations, output_theta)
    elif typ == "dif":
        theta.generate_difficult_thetas_file(permutation_size, num_permutations, output_theta)

    print(f"Generated {num_permutations} permutations, each of size {permutation_size}, at: {output_filepath}")


np.random.seed(42)

number_of_instances = 10
for i in range(number_of_instances):
    generate_permutations_file(20, 10, f"./../data/permutation_easy{i}.txt", 'easy')
    generate_permutations_file(20, 10, f"./../data/permutation_difficult{i}.txt", 'dif')
