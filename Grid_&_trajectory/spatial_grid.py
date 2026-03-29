import json
import cv2

class SpatialGrid:
    def __init__(self, config_path):
        """Initializes the grid using the GITI_grid_config.json structure"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Mapping to your specific JSON keys
        self.corners = self.config["corners"]
        self.cell_size = self.config["configuration"]["cell_size"]
        self.naming_style = self.config["configuration"]["naming_style"]  # "G{col}_{row}"
        
        # Define boundaries for coordinate checking
        self.x_min, self.x_max = self.corners["top_left"][0], self.corners["top_right"][0]
        self.y_min, self.y_max = self.corners["top_left"][1], self.corners["bottom_left"][1]

    def get_cell_from_pixels(self, px_x, px_y):
        """
        Converts (x, y) pixels into a Cell ID (e.g., G_B_3).
        Used to identify where trajectory intersections happen.
        """
        if not (self.x_min <= px_x <= self.x_max and self.y_min <= px_y <= self.y_max):
            return "OUT_OF_BOUNDS"
            
        col_idx = int((px_x - self.x_min) // self.cell_size)
        row_idx = int((px_y - self.y_min) // self.cell_size)
        
        col_letter = chr(65 + (col_idx % 26))
        row_num = row_idx + 1
        
        return self.naming_style.format(col=col_letter, row=row_num)

    def get_cell_center(self, cell_id):
        """
        Reverses a Cell ID back into pixel coordinates.
        Useful for highlighting specific conflict cells on the frame.
        """
        try:
            parts = cell_id.split('_')
            col_letter = parts[1]
            row_num = int(parts[2])
            
            col_idx = ord(col_letter) - 65
            row_idx = row_num - 1
            
            x = self.x_min + (col_idx * self.cell_size) + (self.cell_size // 2)
            y = self.y_min + (row_idx * self.cell_size) + (self.cell_size // 2)
            return (x, y)
        except Exception:
            return None

    def draw_overlay(self, frame, alpha=0.6):
        """
        Renders the high-visibility neon yellow grid with axis labels.
        Designed for clear visibility over dark asphalt.
        """
        overlay = frame.copy()
        line_color = (0, 255, 255)  # Neon Yellow
        shadow = (0, 0, 0)
        
        # 1. Draw Vertical Lines & Column Headers (A, B, C...)
        for i, x in enumerate(range(self.x_min, self.x_max + 1, self.cell_size)):
            cv2.line(overlay, (x, self.y_min), (x, self.y_max), shadow, 3)
            cv2.line(overlay, (x, self.y_min), (x, self.y_max), line_color, 1)
            
            if x < self.x_max:
                label = chr(65 + (i % 26))
                pos = (x + 10, max(30, self.y_min - 15)) 
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow, 4)
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

        # 2. Draw Horizontal Lines & Row Headers (1, 2, 3...)
        for i, y in enumerate(range(self.y_min, self.y_max + 1, self.cell_size)):
            cv2.line(overlay, (self.x_min, y), (self.x_max, y), shadow, 3)
            cv2.line(overlay, (self.x_min, y), (self.x_max, y), line_color, 1)
            
            if y < self.y_max:
                label = str(i + 1)
                pos = (max(5, self.x_min - 45), y + 35)
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow, 4)
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
