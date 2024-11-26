from collections import deque

import cv2


class BallTracker:
    def __init__(self, trajectory_length=10):
        """Initialize the ball tracker"""
        self.trajectory_points = deque(maxlen=trajectory_length)
        self.current_position = None

        # For velocity calculation
        self.prev_y = None
        self.velocity_threshold = 30  # pixels per frame for impact detection

    def update(self, detections):
        """Update ball position and trajectory"""
        self.current_position = None

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            self.current_position = (center_x, center_y)
            self.trajectory_points.append(self.current_position)

            # Update previous y position for velocity calculation
            if self.prev_y is not None:
                dy = center_y - self.prev_y
                if abs(dy) > self.velocity_threshold:
                    # Potential impact detected
                    print(f"Impact detected! dy: {dy}")
            self.prev_y = center_y
            break  # Only process the first detected ball

        if not self.current_position:
            self.trajectory_points.append(None)

    def draw_trajectory(self, frame):
        """Draw ball trajectory on frame"""
        points = list(filter(None, self.trajectory_points))  # Remove None values
        for i in range(1, len(points)):
            alpha = i / len(points)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, points[i - 1], points[i], color, 3)
        return frame

    def get_current_position(self):
        """Return current ball position"""
        return self.current_position