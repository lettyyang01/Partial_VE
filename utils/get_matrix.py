import numpy as np


class Matrix:
    def __init__(self):
        pass

    def get_ei(self, ind, n):
        i=ind
        ei = np.zeros(n)  # Start with a zero vector of length n
        ei[i] = 1
        ei = ei.reshape(-1, 1)
        return ei
        
    def get_Gram(self,X):
        return np.linalg.pinv(X @ X.T)

    def get_Qi(self,ind, W):
        # ind: index of i-th component
        # n: total num of samples
        # W: a wide matrix has full row rank
        
        
        G_W = self.get_Gram(W)
        
        ei = self.get_ei(ind, len(W))
        
        Qi = ei @ ei.T @ G_W / (ei.T @ G_W @ ei)
        
        return Qi

    def get_Hi(self,ind, T, W):
        # ind: which index
        # T: tall matrix full col rank
        # W: wide matrix full row rank
        
        G_W = self.get_Gram(W)
        
        Qi = self.get_Qi(ind, W)
        In = np.identity(len(T))
        Hi = T @ np.linalg.pinv(T.T @ G_W @ (In - Qi) @ T) @ (T.T @ G_W @ (In - Qi))
        
        return Hi
    