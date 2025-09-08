#!/bin/bash

continuos_dimensions=(2 5 10 20 30)
permutations_sizes=(5 10 20 30)
permutation_distances=('K' 'C')
difficulty = ('E' 'H')
permutations_solver_strategies=('n')
instance=('mif', 'qlm')

for run in {1..10}; do
    for cd in "${continuos_dimensions[@]}"; do
        for pm in "${permutations_sizes[@]}"; do
            for d in "${permutation_distances[@]}"; do
                for ps in "${permutations_solver_strategies[@]}"; do
                    for is in "${instance[@]}"; do
                        for dif in "${difficulty[@]}"; do
                            cm=$((cd * 2))
                            sbatch ~/run.sh $cd $cm $pm $d $dif $ps $is
                        done
                    done
                done
            done
        done
    done
done

