#!/bin/bash

continuos_dimensions=(2 3 4)
number_of_continuos_minima=(4 5 7)
permutations_sizes=(4 5 7 10)
permutation_distances=('K' 'C')

for cd in "${continuos_dimensions[@]}"; do
    for cm in "${number_of_continuos_minima[@]}"; do
        for pm in "${permutations_sizes[@]}"; do
            for d in "${permutation_distances[@]}"; do
		        sbatch run.sh $cd $cm $pm $d
            done
        done
    done
done

