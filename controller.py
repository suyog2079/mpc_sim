import numpy as np
import path_gen
import robot

DT = 0.1
P = 20
M = 10

class MPCController:
    def __init__(self, _p, _m, _dt, _A, _B, _C, _Q, _R, _robot, _path):
        """
        initialize MPC controller.
        Args:
            _p (): predistion horizon
            _m (): conrol horizon
            _dt (): time step
            _A (): state transition matrix
            _B (): input matrix 
            _C (): observation matrix
            _robot (): robot instance
            _path (): path to follow
        """
        self.p = _p
        self.m = _m
        self.dt = _dt
        self.A = _A
        self.B = _B
        self.C = _C
        self.robot = _robot
        self.path = _path
        self.Q = _Q
        self.R = _R

    def Sx(self):
        """
        Returns: Sx matrix which represents the effect of the current state on future outputs.
        """
        ns = self.robot.X.shape[0]
        nos = self.C.shape[0]

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
        nos = self.C.shape[0]
        nu = self.B.shape[1]

        S = np.zeros((nos, nu))
        Su_pre = np.zeros((self.p * nos, nu))

        X = self.B

        for i in range(self.p):
            S = self.C @ X
            Su_pre[i * nos:(i + 1) * nos, :] = S
            X = self.A @ X

    def Su(self):
        """
        Returns: Su matrix which represents the effect of future control inputs on future outputs.
        """
        nos = self.C.shape[0]
        nu = self.B.shape[1]

        Su = np.zeros((self.p * nos, self.m * nu))

        for row in range(self.p):
            for col in range(self.m):
                if row >= col:
                    S = self.C @ np.linalg.matrix_power(self.A, row - col) @ self.B
                    Su[row * nos:(row + 1) * nos, col * nu:(col + 1) * nu] = S
        return Su

    def free_response(self, X_k, U_k_prev):
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

    def compute_delta_U(self, F, Y_ref):
        """
        Args:
            F (): free response vector
            Y_ref (): reference output vector

        Returns: optimal change in control input vector delta_U which minimizes the cost function J = ||F + Su * delta_U - Y_ref||^2
        """
        Su = self.Su()
        delta_U = np.linalg.inv(Su.T @ self.Q @ Su + self.R) @ (Su.T @ self.Q @ (Y_ref - F))
        return delta_U

