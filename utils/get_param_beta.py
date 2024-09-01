import numpy as np

class ParamBeta:
    def __init__(self):
        pass

    def get_beta_j_fwl(self, W, T, y):
        In = np.identity(len(T))
        PT_perp = In - T @ np.linalg.pinv(T)
        
        return np.linalg.pinv(PT_perp @ W) @ (PT_perp @ y)

    def get_beta_jc_fwl(self, W,T,y):
        W_dag = np.linalg.pinv(W)
        
        return np.linalg.pinv(W_dag @ T) @ W_dag @ y