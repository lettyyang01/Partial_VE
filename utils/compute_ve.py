import numpy as np
from utils.get_matrix import Matrix
from utils.get_param_beta import ParamBeta

class VarEstimator:
    def __init__(self):
        self.matrix = Matrix()
        self.parambeta = ParamBeta()

    def compute_loo_var_full(self, X, y): 
        G = np.linalg.inv(X@X.T)
        D = np.diag(np.diag(G))
        Prod = np.linalg.inv(D) @ G
        res_loo = Prod @ y 
        tracex = np.trace(Prod @ Prod.T)
        return (np.linalg.norm(res_loo)**2) / tracex 
    
    def compute_loo_var_partial(self, W, T, y): 
        n = len(T)
        G_W = self.matrix.get_Gram(W)
        Diag = np.diag(np.diag(G_W))
        In = np.identity(len(T))
        
        sum_eps = 0
        sum_trace = 0
        for i in range(n):
            ei = self.matrix.get_ei(i, len(T))
            Hi = self.matrix.get_Hi(i, T, W)
            Prod = ei.T @ np.linalg.inv(Diag) @ G_W @ (In - Hi)
            
            # Ensure eps_i is a scalar
            eps_i = float(Prod @ y)
            trace_i = float(np.trace(Prod @ Prod.T))
            
            sum_eps += eps_i**2
            sum_trace += trace_i
            
        return sum_eps / sum_trace

    
    def compute_var_fwl_j(self,W,T,y):
        In = np.identity(len(T))
        PT_perp = In - T @ np.linalg.pinv(T)
        
        res = y - PT_perp @ W @ self.parambeta.get_beta_j_fwl_alt(W, T, y)
        trace = np.trace(In - (PT_perp @ W) @ np.linalg.pinv(PT_perp @ W))
        
        return (np.linalg.norm(res)**2)/trace

    def compute_var_fwl_jc(self,W,T,y):
        W_dag = np.linalg.pinv(W)
        Ij = np.identity(W.shape[1])
        res = W_dag @ y - W_dag @ T @ self.parambeta.get_beta_jc_fwl(W,T,y)
        trace = np.trace((Ij-(W_dag@T)@np.linalg.pinv(W_dag@T)) @ np.linalg.pinv(W.T @ W))
        
        return (np.linalg.norm(res)**2)/trace
    
