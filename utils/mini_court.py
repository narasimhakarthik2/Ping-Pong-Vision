import cv2
import numpy as np
import time
from collections import deque


class TableTennisMiniMap:
    def __init__(self, frame, width=250, height=150):
        # Mini-map dimensions
        self.width = width
        self.height = height
        self.buffer = 50
        self.padding = 5

        # Initialize display areas
        self.setup_display_area(frame)

        # Store perspective transform matrix
        self.perspective_matrix = None

        # Trajectory mask for smooth path
        self.trajectory_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.trajectory_decay = 0.95  # How quickly the trajectory fades

        # Color scheme
        self.colors = {
            'table': (255, 178, 50),
            'lines': (255, 255, 255),
            'net': (255, 0, 0),
            'border': (200, 200, 200),
            'left_side': (65, 105, 225),
            'right_side': (220, 20, 60),
            'shadow': (20, 20, 20),
            'trajectory': (255, 255, 0), # Bright yellow for trajectory
            'impact_left': (0, 255, 0),  # Green for left side impacts
            'impact_right': (0, 0, 255)  # Red for right side impacts
        }

        # Pre-compute the base map
        self.base_map = self.create_base_map()

        # Store last position for smooth interpolation
        self.last_position = None

    def setup_display_area(self, frame):
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        self.start_x = frame_center_x - (self.width // 2)
        self.start_y = self.buffer
        self.end_x = self.start_x + self.width
        self.end_y = self.start_y + self.height

        self.table_start_x = self.start_x + self.padding
        self.table_start_y = self.start_y + self.padding
        self.table_end_x = self.end_x - self.padding
        self.table_end_y = self.end_y - self.padding

        self.table_drawing_width = self.table_end_x - self.table_start_x
        self.table_drawing_height = self.table_end_y - self.table_start_y

        self.net_x = (self.table_start_x + self.table_end_x) // 2
        self.center_y = (self.table_start_y + self.table_end_y) // 2

    def create_base_map(self):
        base = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        table_start_x = self.padding
        table_start_y = self.padding
        table_end_x = self.width - self.padding
        table_end_y = self.height - self.padding
        net_x = self.width // 2
        center_y = self.height // 2

        # Draw table background with gradient
        for y in range(table_start_y, table_end_y):
            alpha = 0.7 + 0.3 * (y - table_start_y) / (table_end_y - table_start_y)
            color = tuple([int(c * alpha) for c in self.colors['table']])
            cv2.line(base, (table_start_x, y), (table_end_x, y), color, 1)

        # Draw border
        cv2.rectangle(base,
                      (table_start_x, table_start_y),
                      (table_end_x, table_end_y),
                      self.colors['border'], 2, cv2.LINE_AA)

        # Draw center line
        cv2.line(base,
                 (table_start_x, center_y),
                 (table_end_x, center_y),
                 self.colors['lines'],
                 1,
                 cv2.LINE_AA)

        # Draw net
        cv2.line(base,
                 (net_x, table_start_y),
                 (net_x, table_end_y),
                 self.colors['net'],
                 4,
                 cv2.LINE_AA)

        return base

    def transform_point_to_minimap(self, point, keypoints):
        """Transform a point from the main view to mini-map coordinates"""
        if keypoints is None or len(keypoints) != 4:
            return None

        if self.perspective_matrix is None:
            src_pts = np.float32([keypoints[0], keypoints[1], keypoints[2], keypoints[3]])
            dst_pts = np.float32([
                [self.table_start_x, self.table_start_y],
                [self.table_end_x, self.table_start_y],
                [self.table_start_x, self.table_end_y],
                [self.table_end_x, self.table_end_y]
            ])
            self.perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        try:
            pt = np.array([[point]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.perspective_matrix)
            x, y = transformed[0][0]

            # Clip coordinates to bounds
            x = int(min(max(x, self.table_start_x), self.table_end_x))
            y = int(min(max(y, self.table_start_y), self.table_end_y))

            return (x, y)
        except Exception:
            return None

    def add_impact(self, position, is_left_side):
        """Add an impact point with timestamp"""
        self.impact_points.append({
            'position': position,
            'timestamp': time.time(),
            'is_left_side': is_left_side
        })

    def draw_impacts(self, frame):
        """Draw impact points with fade effect"""
        current_time = time.time()
        active_impacts = []

        for impact in self.impact_points:
            age = current_time - impact['timestamp']
            if age < self.impact_duration:
                # Calculate fade based on age
                alpha = 1.0 - (age / self.impact_duration)
                color = self.colors['impact_left'] if impact['is_left_side'] else self.colors['impact_right']

                # Draw impact marker with fade
                pos = impact['position']
                color = tuple([int(c * alpha) for c in color])

                # Draw filled circle with outline
                cv2.circle(frame, pos, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pos, 6, (255, 255, 255), 1, cv2.LINE_AA)

                active_impacts.append(impact)

        # Keep only active impacts
        self.impact_points = active_impacts

    def update_trajectory(self, position, is_impact=False):
        """Update the trajectory mask with new ball position"""
        if position is None:
            return

        # Convert position to internal coordinates
        x = position[0] - self.start_x
        y = position[1] - self.start_y

        # Add impact point if needed
        if is_impact:
            self.add_impact((int(x), int(y)), x < self.width // 2)

        # Interpolate between last and current position for smoothness
        if self.last_position is not None:
            # Calculate intermediate points
            steps = 10
            x_steps = np.linspace(self.last_position[0], x, steps)
            y_steps = np.linspace(self.last_position[1], y, steps)

            # Draw smooth line between points
            for i in range(steps):
                pt = (int(x_steps[i]), int(y_steps[i]))
                cv2.circle(self.trajectory_mask, pt, 2, 255, -1)
        else:
            cv2.circle(self.trajectory_mask, (int(x), int(y)), 2, 255, -1)

        self.last_position = (x, y)

        # Apply decay to existing trajectory
        self.trajectory_mask = cv2.multiply(self.trajectory_mask, np.array([self.trajectory_decay]))

        # Blur the trajectory for smoothness
        self.trajectory_mask = cv2.GaussianBlur(self.trajectory_mask, (7, 7), 2)

    def draw_trajectory(self, frame):
        """Draw the smooth trajectory onto the frame"""
        # Create colored trajectory
        colored_trajectory = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(3):
            colored_trajectory[:, :, i] = cv2.multiply(
                self.trajectory_mask,
                np.array([self.colors['trajectory'][i] / 255.0])
            )

        # Add the trajectory to the frame with blending
        frame = cv2.addWeighted(frame, 1.0, colored_trajectory, 0.7, 0)
        return frame

    def draw(self, frame, ball_pos=None, keypoints=None, is_impact=False):
        """Draw the enhanced mini-map with smooth trajectory effect"""
        # Copy the pre-computed base map
        mini_map = self.base_map.copy()

        if ball_pos is not None and keypoints is not None:
            mini_pos = self.transform_point_to_minimap(ball_pos, keypoints)
            if mini_pos is not None:
                # Update and draw trajectory
                self.update_trajectory(mini_pos)
                mini_map = self.draw_trajectory(mini_map)

                # Draw current ball position
                adjusted_pos = (mini_pos[0] - self.start_x, mini_pos[1] - self.start_y)
                is_left_side = mini_pos[0] < self.net_x
                color = self.colors['left_side'] if is_left_side else self.colors['right_side']
                cv2.circle(mini_map, adjusted_pos, 4, color, -1, cv2.LINE_AA)

        # Add shadow effect to mini-map
        shadow = np.zeros((self.height + 4, self.width + 4, 3), dtype=np.uint8)
        shadow[2:2 + self.height, 2:2 + self.width] = mini_map
        shadow = cv2.GaussianBlur(shadow, (5, 5), 0)

        # Copy mini-map to frame
        try:
            frame[self.start_y - 2:self.end_y + 2, self.start_x - 2:self.end_x + 2] = shadow
            frame[self.start_y:self.end_y, self.start_x:self.end_x] = mini_map
        except ValueError as e:
            print(f"Warning: Could not copy mini-map to frame: {e}")
            h, w = mini_map.shape[:2]
            roi_h = self.end_y - self.start_y
            roi_w = self.end_x - self.start_x
            if h != roi_h or w != roi_w:
                mini_map = cv2.resize(mini_map, (roi_w, roi_h))
            frame[self.start_y:self.end_y, self.start_x:self.end_x] = mini_map

        return frame