"""
Simplified social force model baseline for trajectory prediction.

Implements a minimal 2D social force with destination attraction and
repulsive interaction between agents. This is a placeholder for baseline
comparison in research papers.
"""
import numpy as np


class SocialForceModel:
    def __init__(self, dt=0.1, tau=0.5, v0=1.0):
        self.dt = dt
        self.tau = tau
        self.v0 = v0

    def predict(self, positions, velocities, destination, num_steps=10):
        """
        positions: np.array shape (N,2)
        velocities: np.array shape (N,2)
        destination: np.array shape (2,)
        Returns list of positions for each agent.
        """
        positions = np.array(positions, dtype=float)
        velocities = np.array(velocities, dtype=float)
        dest = np.array(destination, dtype=float)
        N = positions.shape[0]

        for _ in range(num_steps):
            desired_dir = dest - positions
            dist = np.linalg.norm(desired_dir, axis=1, keepdims=True)
            e = desired_dir / (dist + 1e-6)
            desired_vel = self.v0 * e
            acceleration = (desired_vel - velocities) / self.tau

            # Simple repulsive force between agents (within 2m)
            for i in range(N):
                for j in range(i+1, N):
                    r_ij = positions[i] - positions[j]
                    d = np.linalg.norm(r_ij)
                    if d < 2.0 and d > 1e-6:
                        force = 1.0 * (1.0 / d - 1.0/2.0) * r_ij / d
                        acceleration[i] += force
                        acceleration[j] -= force

            velocities += acceleration * self.dt
            positions += velocities * self.dt
            yield positions.copy()


if __name__ == "__main__":
    pos = np.array([[0,0],[2,0]])
    vel = np.array([[0.5,0],[-0.5,0]])
    model = SocialForceModel()
    for p in model.predict(pos, vel, np.array([10,0]), 5):
        print(p)
