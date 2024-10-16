import os
import argparse
import numpy as np
import ray
from scipy.linalg import orth

from utils.get_param_beta import ParamBeta

def get_spike(p, n, sigma_x = 1, k = 5, lambda_low = 10, lambda_up = 20):
    #     sigma_x = 1
    #     k = 5 # Number of spikes
    #     lambda_up = 20
    #     lambda_low = 10

    U = np.random.randn(n, p)
    U = orth(U.T).T                              # Orthonormalize the rows of U
    Lambda = np.zeros((p, p))                    # Initialize the spiked covariance matrix
    I_p = np.eye(p)                              # Identity matrix of size p

    for _ in range(k):
        lambda_ell = np.random.uniform(lambda_low, lambda_up)
        v_ell = np.random.randn(p)
        v_ell = v_ell / np.linalg.norm(v_ell)
        Lambda += lambda_ell * np.outer(v_ell, v_ell)
    
    Sigma = sigma_x**2 * (I_p + Lambda)
    sqrt_Sigma = np.linalg.cholesky(Sigma)
    
    W = U @ sqrt_Sigma

    return W


def causal_DGP(n, p, beta_W, beta_0, tau, 
               DGP_W = 'spike', treatment_assign = 'bernoulli'):
    
    if DGP_W == 'spike':
        
        W = get_spike(p-2, n)
    
    Y0 = beta_0 + W @ beta_W
    
    if treatment_assign == 'bernoulli':        
        D = np.random.binomial(1, 0.5, size=n)
        Y1 = Y0 + tau
        y = (1 - D) * Y0 + D * Y1
    elif treatment_assign == 'uniform':
        D = np.random.uniform(0, 1, size=n)
        y = Y0 + D * tau
    
    return W, D.reshape(-1,1), y

@ray.remote
def run_rep(W, D, y_): # single observation on fixed covariates
    n = W.shape[0]
    epsi = np.random.normal(size=n, scale=1)
    y = y_ + epsi
    WT = np.hstack([W, D, np.ones((n, 1))])
    T = np.hstack([D, np.ones((n, 1))])
    pb = ParamBeta()
    
    beta_full = pb.get_full_beta(WT, y)
    beta_partialjc = pb.get_beta_jc_fwl(W, T, y)

    return beta_full[-2], beta_partialjc[0]

@ray.remote
def run_trial(n, p, beta_W, beta_0, tau, rep, treatment):
    tau_full = []
    tau_partial = []
    # Data generation for one trial
    W, D, y_ = causal_DGP(n, p, beta_W, beta_0, tau, treatment_assign = treatment)
    # epsi = np.random.normal(size=n, scale=1)
    # print(np.mean(epsi))

    # y = y_ + epsi
    rep_features = [
        run_rep.remote(W, D, y_) for _ in range(rep)
    ]
    results = ray.get(rep_features)
    for tau_full_rep, tau_partial_rep in results:
        tau_full.append(tau_full_rep)
        tau_partial.append(tau_partial_rep)

    return tau_full, tau_partial


def main():
    ray.init(num_cpus=8, ignore_reinit_error=True)
    treatment = 'bernoulli'

    n = 80
    p = 100
    beta_W = np.ones(p-2) / np.sqrt(p) # np.linspace(0.5, 5, num=p-2)
    beta_0 = 1
    tau_list = [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8]
    tau_list = [element * 10 for element in tau_list]

    trial = 100
    rep = 100

    tau_bias_full = []
    tau_bias_partial = []

    for tau in tau_list:

        trial_futures = [
            run_trial.remote(n, p, beta_W, beta_0, tau, rep, treatment) for _ in range(trial)
        ]
        # Retrieve results
        results = ray.get(trial_futures)
        # Aggregate results
        tau_full = []
        tau_partial = []
        for tau_full_trial, tau_partial_trial in results:
            tau_full.extend(tau_full_trial)  # Use extend to concatenate lists
            tau_partial.extend(tau_partial_trial)
        # Compute biases
        tau_bias_full.append(tau_full)
        tau_bias_partial.append(tau_partial)

    # save the results
    save_dir = 'results/begin_eg'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/tau_bias_full_{treatment}.npy', tau_bias_full)
    np.save(f'{save_dir}/tau_bias_partial_{treatment}.npy', tau_bias_partial)
    
    ray.shutdown()

if __name__ == '__main__':
    main()