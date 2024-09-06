import os
import pandas as pd
import argparse


# experi_type = 'fix_p'
# covar_type = 'spike'
# ve_type = 'full_loo'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run simulations")
    add_arg = parser.add_argument
    add_arg('--ve_type', required=True, 
            help='type of variance estimator')
    add_arg('--experi_type', required=True, 
            help='type of experiment')
    add_arg('--covar_type', required=True, 
            help='type of covariate DGP')
    add_arg('--num_cpus', type=int, default=10,
            help='Specifying number of cpus')
    add_arg('--gpu', type=int, default=None,
            help="Option for local tasks.") # GPU currently not supported
    add_arg('-v', '--verbose', action='store_true',
            help='Enable verbose logging')
    
    return parser.parse_known_args()[0]

def main():
    args = parse_args()

    experi_type = args.experi_type
    covar_type = args.covar_type
    ve_type = args.ve_type

    basic_dir = 'results/'
    file_path = os.path.join(basic_dir, experi_type, ve_type, covar_type)

    files = [f for f in os.listdir(file_path) if f.endswith('.csv')]


    dfs = {}
    print(f'combining {experi_type} files')

    for file in files:
        if experi_type == 'fix_p':
            col_value = file.split('_n')[1].split('_noise')[0]
        elif experi_type == 'fix_ratio':
            col_value = file.split('_p')[-1].split('_n')[0]
        elif experi_type == 'fix_ratio_increasing_noise':
            col_value = file.split('_noise')[2].split('_')[0]
        elif experi_type == 'fix_ratio_increasing_int':
            col_value = file.split('intmag')[1].split('.')[0]

        df = pd.read_csv(os.path.join(file_path, file),header=None)
        dfs[int(col_value)] = df


    combined_df = pd.concat([dfs[key] for key in sorted(dfs)], axis=1)
    combined_df.columns = sorted(dfs.keys())

    save_path = os.path.join(basic_dir, experi_type, f'vedf_{experi_type}_{ve_type}_{covar_type}.csv')
    combined_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
