import cv2
import torch
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Constants
WINDOW_NAME = 'Ping Pong Analyzer'
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FPS_BUFFER_SIZE = 30
CONFIDENCE_THRESHOLD = 0.7
TRIAL_EFFECT_TRAJECTORY_LENGTH = 50


class BallTracker:
    def __init__(self, max_points=100):
        self.trajectory = deque(maxlen=max_points)

    def update(self, results):
        if len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.trajectory.append((center_x, center_y))


class TrailEffect:
    def __init__(self, max_points=100):
        self.points = deque(maxlen=max_points)
        self.min_distance = 100
        self.last_point = None
        # Parameters for glow effect
        self.glow_radius = 31
        self.connection_threshold = 350
        self.node_size = 6
        self.glow_intensity = 0.9

    def update(self, ball_position):
        if ball_position is not None:
            if not self.points:
                self.points.append(ball_position)
                self.last_point = ball_position
                return

            dx = ball_position[0] - self.last_point[0]
            dy = ball_position[1] - self.last_point[1]
            distance = np.sqrt(dx * dx + dy * dy)

            if distance >= self.min_distance:
                self.points.append(ball_position)
                self.last_point = ball_position

    def draw(self, frame):
        if len(self.points) < 2:
            return frame

        # Create separate layers for the glow effect
        base_layer = np.zeros_like(frame, dtype=np.uint8)
        glow_layer = np.zeros_like(frame, dtype=np.uint8)
        points_list = list(self.points)

        # Draw network connections with brightness based on distance
        for i in range(len(points_list)):
            for j in range(i + 1, len(points_list)):
                pt1 = points_list[i]
                pt2 = points_list[j]

                dist = np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
                if dist < self.connection_threshold:
                    # Calculate alpha based on distance and point age
                    alpha = int(255 * (1 - dist / self.connection_threshold) *
                                (1 - i / len(points_list)) * self.glow_intensity)
                    if alpha > 20:
                        # Draw on both base and glow layers
                        cv2.line(base_layer, pt1, pt2, (alpha, alpha, alpha), 2, cv2.LINE_AA)
                        cv2.line(glow_layer, pt1, pt2, (alpha // 2, alpha // 2, alpha // 2), 3, cv2.LINE_AA)

        # Draw nodes with glow
        for i, point in enumerate(points_list):
            alpha = int(255 * (1 - i / len(points_list)))
            if alpha > 20:
                # Main node
                cv2.circle(base_layer, point, self.node_size, (255, 255, 255), -1, cv2.LINE_AA)
                # Outer glow
                cv2.circle(glow_layer, point, self.node_size + 2, (alpha, alpha, alpha), -1, cv2.LINE_AA)
                # Inner bright core
                cv2.circle(base_layer, point, self.node_size - 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Apply gaussian blur to create the glow effect
        glow_layer = cv2.GaussianBlur(glow_layer, (self.glow_radius, self.glow_radius), 0)

        # Blend layers together
        result = cv2.addWeighted(frame, 1, glow_layer, 1, 0)
        result = cv2.addWeighted(result, 1, base_layer, 1, 0)

        return result

class FrameProcessor:
    def __init__(self, max_workers=2):
        self.input_queue = Queue(maxsize=64)
        self.output_queue = Queue(maxsize=64)
        self.stopped = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def start(self):
        threading.Thread(target=self.process_frames, daemon=True).start()
        return self

    def process_frames(self):
        while not self.stopped:
            if not self.input_queue.empty() and not self.output_queue.full():
                task = self.input_queue.get()
                if task is None:
                    continue

                frame, process_fn = task
                try:
                    future = self.executor.submit(process_fn, frame)
                    processed_frame = future.result()
                    self.output_queue.put(processed_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                time.sleep(0.001)

    def stop(self):
        self.stopped = True
        self.executor.shutdown()

class VideoStreamThread:
    def __init__(self, src='Data/game_1.mp4', queue_size=32):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)

        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / self.fps
        self.last_frame_time = 0

        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_buffer = np.zeros((queue_size, self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.buffer_idx = 0

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                current_time = time.time()
                elapsed = current_time - self.last_frame_time

                if elapsed >= self.frame_interval:
                    ret, frame = self.stream.read()
                    if not ret:
                        self.stopped = True
                        break

                    self.frame_buffer[self.buffer_idx] = frame
                    self.queue.put((self.frame_buffer[self.buffer_idx], current_time))
                    self.buffer_idx = (self.buffer_idx + 1) % len(self.frame_buffer)
                    self.last_frame_time = current_time
            else:
                time.sleep(0.001)

    def read(self):
        return self.queue.get()

    def running(self):
        return not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()


class PingPongAnalyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_models()
        self.setup_video()
        self.setup_display()
        self.frame_times = deque(maxlen=FPS_BUFFER_SIZE)
        self.trail_effect = TrailEffect(max_points=30)
        self.frame_processor = FrameProcessor(max_workers=2)
        self.frame_processor.start()
        self.last_frame_time = 0

    def setup_models(self):
        try:
            self.model = YOLO('models/best.pt')
            self.model.to(self.device)
            print(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def setup_video(self):
        self.video_stream = VideoStreamThread()
        self.fps = self.video_stream.fps
        self.frame_width = self.video_stream.frame_width
        self.frame_height = self.video_stream.frame_height
        print(f"Video loaded: {self.fps} FPS")

    def setup_display(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    def process_frame(self, frame, ball_tracker):
        # Create a clean copy for trail overlay
        processed_frame = frame.copy()

        # Ball detection and tracking
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, device=self.device)
        ball_tracker.update(results[0])

        # Update trail effect
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.trail_effect.update((center_x, center_y))

        # Apply trail effect
        processed_frame = self.trail_effect.draw(processed_frame)

        return processed_frame

    def run(self):
        self.video_stream.start()
        ball_tracker = BallTracker(TRIAL_EFFECT_TRAJECTORY_LENGTH)
        frame, _ = self.video_stream.read()

        frame_interval = 1.0 / self.fps

        while self.video_stream.running():
            current_time = time.time()

            # Read frame with timestamp
            frame, frame_timestamp = self.video_stream.read()

            # Process frame
            self.frame_processor.input_queue.put(
                (frame, lambda f: self.process_frame(f, ball_tracker)))

            if not self.frame_processor.output_queue.empty():
                processed_frame = self.frame_processor.output_queue.get()

                # Calculate actual FPS
                elapsed = current_time - self.last_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0

                cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(WINDOW_NAME, processed_frame)

                # Calculate time to wait
                processing_time = time.time() - current_time
                wait_time = max(1, int((frame_interval - processing_time) * 1000))

                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break

                self.last_frame_time = current_time

        self.cleanup()

    def cleanup(self):
        self.video_stream.stop()
        self.frame_processor.stop()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()


def main():
    torch.backends.cudnn.benchmark = True
    try:
        analyzer = PingPongAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()