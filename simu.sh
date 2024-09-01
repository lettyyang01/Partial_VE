#!/bin/bash
#SBATCH --job-name=var_estimation   # Optional: give your job a name
#SBATCH --nodes=1                   # Request 4 nodes
#SBATCH --ntasks-per-node=10        # 10 tasks per node
#SBATCH --cpus-per-task=1           # 1 CPU per task
#SBATCH --mem=10GB                  # Memory per node (ensure this is sufficient)
#SBATCH --time=1:30:00              # Time limit hrs:min:sec
#SBATCH --output=out/slurm_%A.out  # Standard output file
#SBATCH --error=err/slurm_%A.err   # Standard error file
#SBATCH --partition=main           # Partition name (e.g., main, epyc-64, etc.)

module purge
module load gcc/11.3.0 python/3.9.12

# Experiment settings (can be passed as parameters to the Python script)
VE_TYPE='partial_loo'                  # Variance estimator,         Options: 'full_loo', 'partial_loo', 'partial_j', 'partial_jc'
EXPERI_TYPE='fix_p'                    # Experiment/simulation type, Options: 'fix_p', 'fix_ratio', 'fix_ratio_increasing_noise'
COVAR_TYPE='geometric'                 # DGP for X/W,                Options: 'standnorm', 'spike', 'geometric'

# Configuration file and script to run
CONFIG_FILE="configs/var_esti_simu.yaml"
PARAMS_FILE="params/${EXPERI_TYPE}_params.txt"  # Corrected path syntax for Bash
PYTHON_SCRIPT="run_var_esti.py"
WRAP_SCRIPT="combine_dfs.py"

# Initialize a counter
counter=0

# Read each line in the params_file and run two tasks in parallel at a time
while IFS= read -r line
do
    # Run the Python script with specified parameters
    python $PYTHON_SCRIPT --config_file $CONFIG_FILE --ve_type $VE_TYPE --experi_type $EXPERI_TYPE --covar_type $COVAR_TYPE --params "$line" &
    
    counter=$((counter + 1))

    # Check if two tasks have been started
    if [ "$counter" -eq 2 ]; then
        # Wait for both tasks to complete
        wait
        # Reset the counter
        counter=0
    fi
done < "$PARAMS_FILE"

# Wait for any remaining background tasks to complete
wait

python $WRAP_SCRIPT --ve_type $VE_TYPE --experi_type $EXPERI_TYPE --covar_type $COVAR_TYPE
