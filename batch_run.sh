#!/bin/bash

continuos_dimensions=(2 5 10 20 30 50)
permutations_sizes=(5 10 20 30)
permutation_distances=('K' 'C')
permutations_solver_strategies=('n')
instance=('mif')

for run in {1..10}; do
    for cd in "${continuos_dimensions[@]}"; do
        for pm in "${permutations_sizes[@]}"; do
            for d in "${permutation_distances[@]}"; do
                for ps in "${permutations_solver_strategies[@]}"; do
                    for is in "${instance[@]}"; do
                        cm=$((cd * 2))
                        sbatch ~/run.sh $cd $cm $pm $d $ps $is
                    done
                done
            done
        done
    done
done

