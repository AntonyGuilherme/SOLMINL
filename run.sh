#!/bin/bash


#SBATCH --output=output/%j.out
#SBATCH --error=err/%j.err


# echo "Parameter 1: $1"
# echo "Parameter 2: $2"

srun python3 -m src.experiments.runner  $1 $2 $3 $4 $5 $6



