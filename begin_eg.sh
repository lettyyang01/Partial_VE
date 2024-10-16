#!/bin/bash
#SBATCH --job-name=simu_eg   # Optional: give your job a name
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=10        # 10 tasks per node
#SBATCH --cpus-per-task=1           # 1 CPU per task
#SBATCH --mem=10GB                  # Memory per node (ensure this is sufficient)
#SBATCH --time=1:30:00              # Time limit hrs:min:sec
#SBATCH --output=out/slurm_%A.out   # Standard output file
#SBATCH --error=err/slurm_%A.err    # Standard error file
#SBATCH --partition=main            # Partition name (e.g., main, epyc-64, etc.)

module purge
module load gcc/11.3.0 python/3.9.12

PYTHON_SCRIPT="begin_eg.py"

python $PYTHON_SCRIPT