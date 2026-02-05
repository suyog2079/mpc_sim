import numpy as np

class MPC_Controller:
    def __init__(self, _p, _m, _A,  _B, _C, _Qu, _Ru):
        """
        initialize MPC controller.
        Args:
            _p (): prediction horizon
            _m (): control horizon
            _dt (): time step
            _A (): state transition matrix
            _B (): input matrix 
            _C (): observation matrix
            _Qu (): single step stae penelty 
            _Ru (): single step input penalty
        """
        self.p = _p
        self.m = _m
        self.A = _A
        self.B = _B
        self.C = _C
        self.Q = np.kron(np.eye(self.p),_Qu)
        self.R = np.kron(np.eye(self.m),_Ru)

    def Sx(self):
        """
        Returns: Sx matrix which represents the effect of the current state on future outputs.
        """
        # nos = number of states , nu = number of control inputs
        nos = self.C.shape[0]
        ns = self.A.shape[0]

        Sx = np.zeros((self.p * nos, ns))

        S = self.C @ self.A
        for i in range(self.p):
            Sx[i * nos:(i + 1) * nos, :] = S
            S = S @ self.A
        return Sx

    def Su_pre(self):
        """
        Returns: Su_pre matrix which represents the effect of future control inputs on future outputs.
        """
        # nos = number of states , nu = number of control inputs
        nos = self.C.shape[0]
        nu = self.B.shape[1]

        S = np.zeros((nos, nu))
        Su_pre = np.zeros((self.p * nos, nu))

        X = self.B

        for i in range(self.p):
            S = S + self.C @ X
            Su_pre[i * nos:(i + 1) * nos, :] = S
            X = self.A @ X

        return Su_pre

    def Su(self):
        """
        Returns: Su matrix which represents the effect of future control inputs on future outputs.
        """
        # nos = number of states , nu = number of control inputs
        nos = self.C.shape[0]
        nu = self.B.shape[1]

        Su = np.zeros((self.p * nos, self.m * nu))

        for row in range(self.p):
            for col in range(self.m):
                if row >= col:
                    S = self.C @ np.linalg.matrix_power(self.A, row - col) @ self.B
                    Su[row * nos:(row + 1) * nos, col * nu:(col + 1) * nu] = S
        return Su

    def F(self, X_k, U_k_prev):
        """
        Args:
            X_k (): Current state from observer
            U_k_prev (): previouse input vector

        Returns: free response vector F which is the predicted output based on current state and previous inputs and no future input change
        """

        Sx = self.Sx()
        Su_pre = self.Su_pre()

        F = Sx @ X_k + Su_pre @ U_k_prev
        return F

    def delta_U(self,F, Y_ref):
        """
        Args:
            Y_ref (): reference output vector

        Returns: optimal change in control input vector delta_U which minimizes the cost function J = ||F + Su * delta_U - Y_ref||^2
        """
        Su = self.Su()

        H = Su.T @ self.Q @ Su + self.R
        f = Su.T @ self.Q @ (Y_ref - F)
        delta_U = np.linalg.solve(H, f)
        return delta_U

    def control_output(self, X_k, U_k_prev, Y_ref):
        """
        Args:
            X_k: current state estimate
            U_k_prev: previous input (m,)
            Y_ref: reference output over horizon (p*n,)
    
        Returns:
            U_k: optimal control input at time k
        """
        F = self.F(X_k, U_k_prev)  # C.shape[0] * p         
        delta_U = self.delta_U(F, Y_ref)   
    
        m = self.B.shape[1]

        delta_u_k = delta_U[:m]
    
        U_k = U_k_prev + delta_u_k
        return U_k

