import os
import argparse
import numpy as np
import yaml
import ray
from scipy.linalg import orth


from utils.compute_ve import VarEstimator

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
    add_arg('--params', required=True, 
            help='parameters for DGP: p,n,noise_std')
    add_arg('--num_cpus', type=int, default=4,
            help='Specifying number of cpus')
    add_arg('--gpu', type=int, default=None,
            help="Option for local tasks.") # GPU currently not supported
    add_arg('-v', '--verbose', action='store_true',
            help='Enable verbose logging')
    
    return parser.parse_known_args()[0]

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    
    return config 

def get_spike(p, n):
    sigma_x = 1
    k = 5 # Number of spikes
    lambda_up = 20
    lambda_low = 10

    U = np.random.randn(n, p)
    U = orth(U.T).T  # Orthonormalize the rows of U
    Lambda = np.zeros((p, p))  # Initialize the spiked covariance matrix
    I_p = np.eye(p)            # Identity matrix of size p

    for _ in range(k):
        lambda_ell = np.random.uniform(lambda_low, lambda_up)
        v_ell = np.random.randn(p)
        v_ell = v_ell / np.linalg.norm(v_ell)
        Lambda += lambda_ell * np.outer(v_ell, v_ell)
    
    Sigma = sigma_x**2 * (I_p + Lambda)
    sqrt_Sigma = np.linalg.cholesky(Sigma)
    
    W = U @ sqrt_Sigma

    return W

def get_geometric(p, n):
    lambda_ = 1
    rho = 0.95

    U = np.random.randn(n, p)
    U = orth(U.T).T
    V = np.random.randn(p, p)
    V = orth(V.T).T

    # Generate Σ = λ^2 * diag(ρ^ℓ)
    lambdas = np.array([rho**i for i in range(p)])
    Sigma = lambda_**2 * np.diag(lambdas)

    # Calculate W = UΣ^(1/2)V⊤
    W = U @ np.sqrt(Sigma) @ V.T

    return W



def covar_DGP(covar_type, p, n):
    if covar_type == 'spike':
        W = get_spike(p, n)

    elif covar_type == 'standnorm':
        signal_std = 1                                           #PARAMETER: For now, signal_std is fixed to 1
        W = np.random.normal(size=(n, p), scale=signal_std)

    elif covar_type == 'geometric':
        W = get_geometric(p, n)     

    return W

def DGP_partial(covar_type, p, n, beta, beta_0):

    W = covar_DGP(covar_type, p-1, n)                             # ALWAYS p-1, one column for constant
    T = np.ones(n).reshape(-1, 1)  #D.reshape(-1, 1), 
    Xbeta = W @ beta[:(p-1)] + beta_0
       
    return W, T, Xbeta

@ray.remote
def simu_batchX(ve_type, covar_type, p, n, beta, beta_0, noise_std, iter):
    
    W, T, Xbeta = DGP_partial(covar_type, p, n, beta, beta_0)       
    estimates = [worker_compute_ve.remote(ve_type, W, T, Xbeta, noise_std) for _ in range(iter)]
    
    var_esti_list = []
    
    for result in ray.get(estimates):
        var_esti_list.append(result)

    return var_esti_list

@ray.remote
def worker_compute_ve(ve_type, W, T, Xbeta, noise_std):
    X = np.hstack([W, T])
    y = Xbeta + np.random.normal(size=len(W), scale=noise_std)

    ve = VarEstimator()

    if ve_type == 'full_loo':
        var_esti = ve.compute_loo_var_full(X, y)
        return var_esti
    elif ve_type == 'partial_loo':
        var_esti = ve.compute_loo_var_partial(W, T, y)
        return var_esti
    elif ve_type == 'partial_w':
        var_esti = ve.compute_var_fwl_w(W, T, y)
        return var_esti
    elif ve_type == 'partial_wc':
        var_esti = ve.compute_var_fwl_wc(W, T, y)
        return var_esti


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    iter = config['iterations'] # iteration for each X

    experi_type = args.experi_type
    covar_type = args.covar_type
    ve_type = args.ve_type
    params = args.params.split(' ')
    p = int(params[0])
    n = int(params[1])
    noise_std = int(params[2])
    intercept_mag = int(params[3])

    output_dir = config['output_dir']
    output_dir = os.path.join(output_dir, experi_type, ve_type, covar_type)
    os.makedirs(output_dir, exist_ok=True)  

    beta = np.ones(p) / np.sqrt(p)
    beta_0 = intercept_mag

    print("=====================")
    print("*** Simulation start ***")
    print("=====================")


    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)
    batches_X = config['batches_X']  # Adjust this based on your needs
    results = ray.get([simu_batchX.remote(ve_type, covar_type, p, n, beta, beta_0, noise_std, iter) for _ in range(batches_X)])
    
    all_var_estimates = []

    for result in results:
        all_var_estimates.extend(result)  # var_esti_list from each batch

    output_file = os.path.join(output_dir, f'vedf_{experi_type}_{ve_type}_{covar_type}_p{p}_n{n}_noise{noise_std}_intmag{intercept_mag}.csv')
    with open(output_file, 'w') as f:
        f.write("var_esti\n")  # Write header with a newline

        # Write each estimate on a new line
        for var_esti in all_var_estimates:
            f.write(f"{var_esti}\n")
    
    # Shutdown Ray
    ray.shutdown()

    print("=====================")
    print("*** Results saved ***")
    print("=====================")

    

if __name__ == '__main__':
    main()
