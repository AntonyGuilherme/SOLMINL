#!/bin/bash

continuos_dimensions=(2 3 4)
number_of_continuos_minima=(4 5 7)
permutations_sizes=(5 7 10)
permutation_distances=('K' 'C')
permutations_solver_strategies=('n' 'nsc' 'nsi')
instance=('mif')

for cd in "${continuos_dimensions[@]}"; do
    for cm in "${number_of_continuos_minima[@]}"; do
        for pm in "${permutations_sizes[@]}"; do
            for d in "${permutation_distances[@]}"; do
                for ps in "${permutations_solver_strategies[@]}"; do
                    for is in "${instance[@]}"; do
		                sbatch ~/run.sh $cd $cm $pm $d $ps $is
                    done
                done
            done
        done
    done
done

