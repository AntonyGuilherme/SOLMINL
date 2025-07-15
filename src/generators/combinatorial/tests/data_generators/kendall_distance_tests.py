import os
from scipy.stats import kendalltau # Needed for Kendall-Tau calculations

def calculate_kendall_tau_distances(permutations):
    """
    Calculates the Kendall-Tau distance (number of discordant pairs)
    between a reference permutation and a list of other permutations.

    Args:
        permutations: A list where the first sub-list is the
                      reference permutation, and subsequent
                      sub-lists are the permutations to compare against.
    Returns:
        list: A list of Kendall-Tau distances to the first permutation.
    """
    reference_permutation = permutations[0]
    distances = []
    n = len(reference_permutation)

    # The maximum possible number of pairs for a permutation of length n
    max_possible_pairs = n * (n - 1) / 2

    # Iterate through the permutations starting from the second one
    for current_permutation in permutations:
        # Calculate the Kendall tau correlation coefficient
        tau, _ = kendalltau(reference_permutation, current_permutation)

        # Convert the Kendall tau coefficient to the number of discordant pairs (distance)
        # Formula: discordant_pairs = (1 - tau) * (N * (N - 1) / 2) / 2
        discordant_pairs = (1 - tau) * max_possible_pairs / 2
        distances.append(round(discordant_pairs)) # Round to nearest integer

    return distances


def generate_kendall_distances_file(permutations_list, output_filepath):
    """
    Calculates Kendall-Tau distances from a list of permutations (using the first as reference)
    and writes them to a file in a single row, space-separated.

    Args:
        permutations_list (list of list): A list of permutations. The first permutation
                                          will be used as the reference.
        output_filepath (str): The path to the output file.
    """

    distances = calculate_kendall_tau_distances(permutations_list)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    with open(output_filepath, 'w') as f:
        f.write(' '.join(map(str, distances)) + '\n')

    print(f"Generated Kendall-Tau distances file at: {output_filepath}")