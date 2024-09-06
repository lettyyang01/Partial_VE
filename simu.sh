#!/bin/bash
#SBATCH --job-name=var_estimation   # Optional: give your job a name
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=10        # 10 tasks per node
#SBATCH --cpus-per-task=1           # 1 CPU per task
#SBATCH --mem=10GB                  # Memory per node (ensure this is sufficient)
#SBATCH --time=8:30:00              # Time limit hrs:min:sec
#SBATCH --output=out/slurm_%A.out   # Standard output file
#SBATCH --error=err/slurm_%A.err    # Standard error file
#SBATCH --partition=main            # Partition name (e.g., main, epyc-64, etc.)

module purge
module load gcc/11.3.0 python/3.9.12

# Experiment settings
VE_TYPE_LIST=("partial_j") #"full_loo" "partial_jc" "partial_loo"
EXPERI_TYPE_LIST=('fix_p')    # 'fix_ratio' 'fix_ratio_increasing_noise' 'fix_ratio_increasing_int'
COVAR_TYPE_LIST=('standnorm') #  'geometric' 'spike'

# Configuration file and script to run
CONFIG_FILE="configs/var_esti_simu.yaml"
PYTHON_SCRIPT="run_var_esti.py"
WRAP_SCRIPT="combine_dfs.py"

# Loop over every combination of VE_TYPE, EXPERI_TYPE, and COVAR_TYPE
for VE_TYPE in "${VE_TYPE_LIST[@]}"; do
    for EXPERI_TYPE in "${EXPERI_TYPE_LIST[@]}"; do
        for COVAR_TYPE in "${COVAR_TYPE_LIST[@]}"; do

            echo "Running for VE_TYPE: $VE_TYPE, EXPERI_TYPE: $EXPERI_TYPE, COVAR_TYPE: $COVAR_TYPE"

            # Define the correct params file for the current experiment type
            PARAMS_FILE="params/${EXPERI_TYPE}_params.txt"

            # Check if the parameter file exists
            if [[ -f "$PARAMS_FILE" ]]; then
                # Read each line in the params_file
                while IFS= read -r line; do
                    # Run the Python script with specified parameters
                    python $PYTHON_SCRIPT --config_file $CONFIG_FILE --ve_type $VE_TYPE --experi_type $EXPERI_TYPE --covar_type $COVAR_TYPE --params "$line"
                done < "$PARAMS_FILE"

                # Run the wrap-up script after all tasks are done for the current combination
                python $WRAP_SCRIPT --ve_type $VE_TYPE --experi_type $EXPERI_TYPE --covar_type $COVAR_TYPE
            else
                echo "Parameter file $PARAMS_FILE does not exist. Skipping..."
            fi

        done
    done
done
