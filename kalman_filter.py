import numpy as np

class KalmanFilter:
    def __init__(self, _A, _B, _C, _X, _P, _Q, _R):
        """

        Args:
            _A (): State transition matrix
            _B (): Control input matrix
            _C (): Measurement matrix
            _X (): Initial state estimate
            _P (): state covariance
            _Q (): Process noise
            _R (): Measurement noise
        """
        self.A = _A  
        self.B = _B  
        self.C = _C  
        self.X = _X
        self.P = _P  
        self.Q = _Q 
        self.R = _R  
        self.x = self.X.copy()

    def predict(self, U):
        self.x = self.A @ self.X + self.B @ U
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, Z):
        Y = Z - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.X = self.x + K @ Y
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P

    def state_estimate(self):
        return self.X
