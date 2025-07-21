from generators.combinatorial.instance_generator import Permutation, _create_permutations
import time
import numpy as np

import matplotlib.pyplot as plt

sizes = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
normal_times = []
bucket_times = []

for permutation_size in sizes:
    permutation = Permutation(permutation_size, permutation_size**2)
    permutation.calc_parameters_difficult()

    # Time normal evaluate
    start = time.process_time_ns()
    for consensus in permutation.consensus:
        value = permutation.evaluate(consensus)
    end = time.process_time_ns()
    normal_time = (end - start) / np.pow(10, 9)
    normal_times.append(normal_time)

    # Time bucket_evaluate
    start = time.process_time_ns()
    for consensus in permutation.consensus:
        value = permutation.bucket_evaluate(consensus)
    end = time.process_time_ns()
    bucket_time = (end - start) / np.pow(10, 9)
    bucket_times.append(bucket_time)

    print(f"Size {permutation_size}: normal {normal_time:.4f}s, bucket {bucket_time:.4f}s")

# Plotting
plt.plot(sizes, normal_times, label='Normal Evaluate')
plt.plot(sizes, bucket_times, label='Bucket Evaluate')
plt.xlabel('Permutation Size')
plt.ylabel('Time (seconds)')
plt.title('Evaluation Time vs Permutation Size')
plt.legend()
plt.grid(True)
plt.savefig("evaluation_times.png")
plt.savefig("time.png")

# print(f"([1,2,3], [3,2,1]) {permutation.hamming([1,2,3], [3,2,1])}")
# print(f"([1,2,3], [1,2,3]) {permutation.hamming([1,2,3], [1,2,3])}")
# print(f"([1,2,3], [2,1,3]) {permutation.hamming([1,2,3], [2,1,3])}")






