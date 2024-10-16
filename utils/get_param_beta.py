import numpy as np

class ParamBeta:
    def __init__(self):
        pass

    def get_full_beta(self, X, y):
    # return the [beta] for y=X*\beta
        pinv = np.linalg.pinv(X)
        return pinv @ y
    
    def get_beta_j_fwl(self, W, T, y):
        In = np.identity(len(T))
        PT_perp = In - T @ np.linalg.pinv(T)
        
        return np.linalg.pinv(PT_perp @ W) @ (PT_perp @ y)
    
    def get_beta_j_fwl_alt(self, W, T, y):
        P1 = W.T @ np.linalg.pinv(W.T)
        
        Wdag = np.linalg.pinv(W)
        WdagT = Wdag @ T
        In = np.identity(len(WdagT))
        P2 = In - WdagT @ np.linalg.pinv(WdagT)

        return P1 @ P2 @ Wdag @ y

    def get_beta_jc_fwl(self, W,T,y):
        W_dag = np.linalg.pinv(W)
        
        return np.linalg.pinv(W_dag @ T) @ W_dag @ y