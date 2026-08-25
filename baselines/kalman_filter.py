"""
Simple Kalman filter baseline for trajectory filtering.

This is a placeholder to demonstrate baseline inclusion.
"""
import numpy as np


class SimpleKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()
