import numpy as np

class BEVMapper:
    def __init__(self, H_pixel_to_world, bev_bounds, bev_resolution):
        self.H = np.asarray(H_pixel_to_world, dtype=np.float32)

        self.bev_x_min = float(bev_bounds["x_min"])
        self.bev_x_max = float(bev_bounds["x_max"])
        self.bev_y_min = float(bev_bounds["y_min"])
        self.bev_y_max = float(bev_bounds["y_max"])

        self.bev_w, self.bev_h = map(int, bev_resolution)
        self.mpp_x = (self.bev_x_max - self.bev_x_min) / self.bev_w
        self.mpp_y = (self.bev_y_max - self.bev_y_min) / self.bev_h

    def pixel_to_world(self, p):
        x, y = p
        v = np.array([x, y, 1.0], dtype=np.float32)
        w = self.H @ v
        if w[2] == 0:
            return None
        w /= w[2]
        return float(w[0]), float(w[1])

    def world_to_bev(self, world_xy):
        X, Y = world_xy
        u = int((X - self.bev_x_min) / self.mpp_x)
        v = int((Y - self.bev_y_min) / self.mpp_y)
        return u, v
