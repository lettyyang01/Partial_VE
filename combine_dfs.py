import os
import pandas as pd
import argparse
import yaml

# experi_type = 'fix_p'
# covar_type = 'spike'
# ve_type = 'full_loo'


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    
    return config 

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run simulations")
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/var_esti_simu.yaml',
        help='YAML configuration file')
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
    config = load_config(args.config)

    experi_type = args.experi_type
    covar_type = args.covar_type
    ve_type = args.ve_type

    basic_dir = config['output_dir']
    file_path = os.path.join(basic_dir, experi_type, ve_type, covar_type)

    files = [f for f in os.listdir(file_path) if f.endswith('.csv')]


    dfs_var_esti = {}
    dfs_nums = {}
    dfs_doms = {}
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

        
        dfs_var_esti[int(col_value)] = df.iloc[1:, 0]  # First column for var_esti
        dfs_nums[int(col_value)] = df.iloc[1:, 1]      # Second column for nums
        dfs_doms[int(col_value)] = df.iloc[1:, 2]      # Third column for doms


    combined_var_esti_df = pd.concat([dfs_var_esti[key] for key in sorted(dfs_var_esti)], axis=1)
    combined_var_esti_df.columns = sorted(dfs_var_esti.keys())

    combined_nums_df = pd.concat([dfs_nums[key] for key in sorted(dfs_nums)], axis=1)
    combined_nums_df.columns = sorted(dfs_nums.keys())

    combined_doms_df = pd.concat([dfs_doms[key] for key in sorted(dfs_doms)], axis=1)
    combined_doms_df.columns = sorted(dfs_doms.keys())

    save_path_varesti = os.path.join(basic_dir, experi_type,'var', f'{experi_type}_{ve_type}_{covar_type}_varesti.csv')
    save_path_num = os.path.join(basic_dir, experi_type, ve_type, covar_type, 'num', f'{experi_type}_{ve_type}_{covar_type}_num.csv')
    save_path_dom = os.path.join(basic_dir, experi_type, ve_type, covar_type, 'dom', f'{experi_type}_{ve_type}_{covar_type}_dom.csv')
    os.makedirs(os.path.dirname(save_path_varesti), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_num), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_dom), exist_ok=True)

    combined_var_esti_df.to_csv(save_path_varesti, index=False)
    combined_nums_df.to_csv(save_path_num, index=False)
    combined_doms_df.to_csv(save_path_dom, index=False)



if __name__ == '__main__':
    main()
