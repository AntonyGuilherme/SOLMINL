import math
import numpy as np
import random
import os
import re # For regular expressions to parse filenames
from pyomo.environ import *
from ..parameters.generateCOP import generateCOP

# --- File Comparison Utility (from test_bench.py) ---
def compare_files(file1_path, file2_path, tolerance=1e-3):
    """
    Compares two text files line by line, handling floating point differences
    within a given tolerance.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.
        tolerance (float): Absolute tolerance for comparing float values.

    Returns:
        bool: True if files are identical (within tolerance for floats), False otherwise.
        list: A list of strings describing differences found.
    """
    differences = []
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            if len(lines1) != len(lines2):
                differences.append(f"Number of lines differ: {len(lines1)} vs {len(lines2)}")
                # Continue comparison for common lines
            
            num_lines_to_compare = min(len(lines1), len(lines2))

            for i in range(num_lines_to_compare):
                line1_parts = lines1[i].strip().split()
                line2_parts = lines2[i].strip().split()

                if len(line1_parts) != len(line2_parts):
                    differences.append(f"Line {i+1}: Number of elements differ. File1: {len(line1_parts)}, File2: {len(line2_parts)}")
                    # Continue comparing common elements
                
                num_elements_to_compare = min(len(line1_parts), len(line2_parts))

                line_diff_found = False
                for j in range(num_elements_to_compare):
                    val1_str = line1_parts[j]
                    val2_str = line2_parts[j]

                    try:
                        val1_float = float(val1_str)
                        val2_float = float(val2_str)
                        if abs(val1_float - val2_float) > tolerance:
                            differences.append(f"Line {i+1}, Element {j+1}: Float values differ significantly. File1: {val1_float}, File2: {val2_float}")
                            line_diff_found = True
                    except ValueError:
                        # Not a float, compare as string
                        if val1_str != val2_str:
                            differences.append(f"Line {i+1}, Element {j+1}: String values differ. File1: '{val1_str}', File2: '{val2_str}'")
                            line_diff_found = True
                
                if line_diff_found:
                    differences.append(f"Line {i+1} (File1): {lines1[i].strip()}")
                    differences.append(f"Line {i+1} (File2): {lines2[i].strip()}")

            if not differences:
                return True, []
            else:
                return False, differences

    except FileNotFoundError as e:
        return False, [f"Error: One of the files not found - {e}"]
    except Exception as e:
        return False, [f"An unexpected error occurred during file comparison: {e}"]

# --- Main Batch Processing Logic (Python version of run_generateCOP_batch.R) ---
if __name__ == '__main__':
    print("--- Starting Python Batch Processing ---")

    # Define fixed parameters
    N_PARAM = 10
    M_PARAM = 20
    G_PARAM = "max"
    TYPE_DISTANCE_PARAM = "K"

    # Define the folder where your input files are located
    # IMPORTANT: Adjust this path to the actual folder containing your files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_files_folder = os.path.join(script_dir, "data", "source")
    output_dir = os.path.join(script_dir, "data", "outputs")
    expected_output_dir = os.path.join(script_dir, "data", "expected_outputs")
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(expected_output_dir, exist_ok=True) # Ensure this exists and contains R's outputs

    # --- Dynamic determination of instance types and numbers ---
    all_files = os.listdir(input_files_folder)

    # Regex to capture type (easy, difficult, similar) and number
    permutation_file_pattern = re.compile(r"^permutation_(easy|difficult|similar)([0-9]+)\.txt$")

    unique_instances = []
    for filename in all_files:
        print(filename)
        match = permutation_file_pattern.match(filename)
        if match:
            instance_type = match.group(1)
            instance_number = int(match.group(2))
            if (instance_type, instance_number) not in unique_instances:
                unique_instances.append((instance_type, instance_number))

    # Sort the instances for consistent processing order
    unique_instances.sort()

    if not unique_instances:
        print(f"No 'permutation_TYPE_X.txt' files (easy, difficult, similar) found in {input_files_folder}.\n")
        print("Please ensure input files are in the correct folder.")
    else:
        print(f"Found {len(unique_instances)} unique instances to process.")
        print("Processing instances:")
        for inst_type, inst_num in unique_instances:
            print(f"  - Type: {inst_type}, Number: {inst_num}")
        print("\n")

        # Loop through the dynamically determined instance types and numbers
        for current_type, current_number in unique_instances:
            # Construct the input file paths
            file_sigma = os.path.join(input_files_folder, f"permutation_{current_type}{current_number}.txt")
            file_distances = os.path.join(input_files_folder, f"permutation_{current_type}{current_number}_distance.txt")
            file_theta = os.path.join(input_files_folder, f"permutation_{current_type}{current_number}_theta.txt")
            
            # Construct the output file path for Python's result
            python_output_file = os.path.join(output_dir, f"{current_type}{current_number}.txt")
            
            # Construct the expected output file path (from R's run)
            # Assuming R's output names are like vasco_TYPE_X.txt in the expected_output_dir
            r_expected_output_file = os.path.join(expected_output_dir, f"{current_type}{current_number}.txt")
            
            # Call the generateCOP function
            if os.path.exists(file_sigma) and os.path.exists(file_distances) and os.path.exists(file_theta):
                if current_type == "similar":
                    G_PARAM = "sim"
                elif current_type == "difficult":
                    G_PARAM = "min"
                    
                success = generateCOP(n=N_PARAM, 
                                      m=M_PARAM, 
                                      FileSigma=file_sigma, 
                                      FileDistances=file_distances, 
                                      FileTheta=file_theta, 
                                      G=G_PARAM, 
                                      typeDistance=TYPE_DISTANCE_PARAM, 
                                      FileOut=python_output_file)
                
                if success:                    
                    # Compare outputs
                    if os.path.exists(r_expected_output_file):
                        tolerance = 1e-2
                        is_match, diffs = compare_files(python_output_file, r_expected_output_file, tolerance)
                        
                        if is_match:
                            print(f"\033[92mComparison Result: PASSED - Outputs match! {current_type} {current_number}\033[0m")
                        else:
                            print(f"\033[91mComparison Result: FAILED - Outputs DO NOT match({tolerance}). {current_type} {current_number}\033[0m")
                    else:
                        print(f"WARNING: R's expected output file '{r_expected_output_file}' not found. Cannot compare.")
                        print(f"Please ensure you have run the R script to generate expected outputs in '{expected_output_dir}'.")
                else:
                    print(f"Python generateCOP failed for instance Type: {current_type}, Number: {current_number}. Skipping comparison.")
            else:
                print(f"Skipping instance Type: {current_type}, Number: {current_number}: One or more input files missing.")
                print(f"  Missing: Sigma ({os.path.exists(file_sigma)}), Distances ({os.path.exists(file_distances)}), Theta ({os.path.exists(file_theta)})")

    print("--- Python Batch Processing Complete ---")