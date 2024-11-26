import cv2
import torch
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
from SAM.model import UNet
from utils.mini_court import TableTennisMiniMap
from config.settings import *
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from table_key_points.model import KeypointModel


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
    def __init__(self, src='Data/quad.mp4', queue_size=32):
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


class QuadViewProcessor:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.quad_width = frame_width // 2
        self.quad_height = frame_height // 2

    def create_quad_layout(self, original_frame, ball_tracking_frame,
                           table_keypoints_frame, ball_segmentation_frame):
        # Create a blank canvas
        layout = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Resize all frames to fit quadrants
        original_resized = cv2.resize(original_frame, (self.quad_width, self.quad_height))
        ball_tracking_resized = cv2.resize(ball_tracking_frame, (self.quad_width, self.quad_height))
        table_keypoints_resized = cv2.resize(table_keypoints_frame, (self.quad_width, self.quad_height))
        ball_segmentation_resized = cv2.resize(ball_segmentation_frame, (self.quad_width, self.quad_height))

        # Place frames in their respective quadrants
        layout[0:self.quad_height, 0:self.quad_width] = original_resized  # Top-left
        layout[0:self.quad_height, self.quad_width:self.frame_width] = ball_tracking_resized  # Top-right
        layout[self.quad_height:self.frame_height, 0:self.quad_width] = table_keypoints_resized  # Bottom-left
        layout[self.quad_height:self.frame_height,
        self.quad_width:self.frame_width] = ball_segmentation_resized  # Bottom-right

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (255, 255, 255)

        cv2.putText(layout, "Original Video", (10, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(layout, "Ball Tracking", (self.quad_width + 10, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(layout, "Table Segmentation + Key points", (10, self.quad_height + 30), font, font_scale, font_color,
                    font_thickness)
        cv2.putText(layout, "Ball track + Segmentation", (self.quad_width + 10, self.quad_height + 30), font, font_scale,
                    font_color, font_thickness)

        return layout


class BallTracker:
    def __init__(self, max_trajectory_length=5):
        self.trajectory = deque(maxlen=max_trajectory_length)
        self.current_position = None

    def update(self, detection_results):
        boxes = detection_results.boxes
        if len(boxes) > 0:
            # Get the box with highest confidence
            conf = boxes.conf.cpu().numpy()
            if len(conf) > 0:
                best_idx = np.argmax(conf)
                box = boxes[best_idx]
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                self.current_position = (center_x, center_y)
                self.trajectory.append(self.current_position)
        else:
            self.current_position = None
            self.trajectory.append(None)

    def draw_trajectory(self, frame):
        # Draw trajectory
        for i in range(1, len(self.trajectory)):
            if self.trajectory[i] is not None and self.trajectory[i - 1] is not None:
                cv2.line(frame, self.trajectory[i - 1], self.trajectory[i], (0, 255, 0), 3)

        # Draw current position
        if self.current_position:
            cv2.circle(frame, self.current_position, 0, (0, 0, 255), -1)

        return frame

    def get_current_position(self):
        return self.current_position

    def get_trajectory(self):
        return list(self.trajectory)


class PingPongAnalyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_models()
        self.setup_video()
        self.setup_display()
        self.frame_times = deque(maxlen=FPS_BUFFER_SIZE)
        self.cached_mask = None
        self.segmentation_interval = 120
        self.frame_count = 0
        self.stored_keypoints = None
        self.last_frame_time = 0

        self.quad_processor = QuadViewProcessor(self.frame_width, self.frame_height)
        self.frame_processor = FrameProcessor(max_workers=2)
        self.frame_processor.start()

    def setup_models(self):
        try:
            # Load YOLO model
            self.model = YOLO('models/best.pt')
            self.model.to(self.device)
            print(f"YOLO model loaded successfully on {self.device}")

            # Load table segmentation model
            self.table_segmentation_model = UNet()
            self.table_segmentation_model.load_state_dict(
                torch.load('SAM/output/table_segmentation_model.pth', map_location=self.device))
            self.table_segmentation_model.to(self.device).eval()
            print(f"Table segmentation model loaded successfully on {self.device}")

            # Load keypoint detection model
            self.keypoint_model = KeypointModel(num_keypoints=4)
            self.keypoint_model.load_state_dict(
                torch.load('table_key_points/saved_models/best_model.pth', map_location=self.device)[
                    'model_state_dict']
            )
            self.keypoint_model.to(self.device).eval()
            print(f"Keypoint detection model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

    def detect_keypoints(self, frame):
        if self.stored_keypoints is not None:
            return self.stored_keypoints

        with torch.no_grad():
            image = cv2.resize(frame, (224, 224))
            image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            image = image.to(self.device)

            keypoints = self.keypoint_model(image)
            keypoints = keypoints.cpu().numpy().reshape(-1, 2)

            scale_x = frame.shape[1] / 224
            scale_y = frame.shape[0] / 224
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

            # Apply offsets
            offset_x = 80
            offset_y = 10
            keypoints[:, 0] += offset_x
            keypoints[:, 1] += offset_y

            # Add extra adjustment to left top point
            left_top_x_offset = -20
            keypoints[0, 0] += left_top_x_offset

            self.stored_keypoints = keypoints
            self.stored_keypoints = keypoints
            return keypoints

    def create_ball_segmentation_view(self, frame, ball_tracker, segmentation_mask):
        black_bg = np.zeros_like(frame)

        current_pos = ball_tracker.get_current_position()
        if current_pos:
            cv2.circle(black_bg, (int(current_pos[0]), int(current_pos[1])), 8, (255, 255, 255), -1)

            trajectory = ball_tracker.get_trajectory()
            for i in range(1, len(trajectory)):
                if trajectory[i] is not None and trajectory[i - 1] is not None:
                    pt1 = tuple(map(int, trajectory[i - 1]))
                    pt2 = tuple(map(int, trajectory[i]))
                    cv2.line(black_bg, pt1, pt2, (0, 255, 0), 3)

        if segmentation_mask is not None:
            mask_overlay = np.zeros_like(frame)
            mask_overlay[segmentation_mask > 0.5] = [0, 0, 255]
            black_bg = cv2.addWeighted(black_bg, 1.0, mask_overlay, 0.5, 0)

        return black_bg

    def create_table_keypoints_view(self, frame, keypoints, segmentation_mask):
        table_view = np.zeros_like(frame)
        if segmentation_mask is not None:
            table_view[segmentation_mask > 0.5] = [0, 255, 0]

        if keypoints is not None:
            corner_indices = [0, 1, 3, 2, 0]
            for i in range(len(corner_indices) - 1):
                pt1 = tuple(map(int, keypoints[corner_indices[i]]))
                pt2 = tuple(map(int, keypoints[corner_indices[i + 1]]))
                cv2.line(table_view, pt1, pt2, (0, 0, 255), 5, cv2.LINE_AA)

            for point in keypoints:
                pt = tuple(map(int, point))
                cv2.circle(table_view, pt, 5, (255, 0, 0), -1)

        return table_view

    def process_frame(self, frame, ball_tracker, mini_map):
        original_frame = frame.copy()

        if self.cached_mask is None or self.frame_count % self.segmentation_interval == 0:
            with torch.no_grad():
                input_frame = cv2.resize(frame, (256, 256))
                input_tensor = torch.from_numpy(input_frame).permute(2, 0, 1).unsqueeze(0).float().to(
                    self.device) / 255.0
                segmentation_mask = self.table_segmentation_model(input_tensor)
                segmentation_mask = segmentation_mask.squeeze().cpu().numpy()
                self.cached_mask = cv2.resize(segmentation_mask, (frame.shape[1], frame.shape[0]))

        self.frame_count += 1

        if self.stored_keypoints is None:
            keypoints = self.detect_keypoints(frame)
        else:
            keypoints = self.stored_keypoints

        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, device=self.device)
        ball_tracker.update(results[0])

        ball_tracking_frame = frame.copy()
        ball_tracking_frame = ball_tracker.draw_trajectory(ball_tracking_frame)

        ball_segmentation_frame = self.create_ball_segmentation_view(
            frame, ball_tracker, self.cached_mask)

        table_keypoints_frame = self.create_table_keypoints_view(
            frame, keypoints, self.cached_mask)

        quad_view = self.quad_processor.create_quad_layout(
            original_frame,  # Top-left
            ball_tracking_frame,  # Top-right
            table_keypoints_frame,  # Bottom-left (Table Segmentation)
            ball_segmentation_frame  # Bottom-right (Ball Segmentation)
        )

        return quad_view

    def setup_video(self):
        self.video_stream = VideoStreamThread()
        self.fps = self.video_stream.fps
        self.frame_width = self.video_stream.frame_width
        self.frame_height = self.video_stream.frame_height
        print(f"Video loaded: {self.fps} FPS")

    def setup_display(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    def run(self):
            self.video_stream.start()
            ball_tracker = BallTracker(5)
            frame, _ = self.video_stream.read()

            mini_map = TableTennisMiniMap(
                frame,
                width=MINI_MAP_WIDTH,
                height=MINI_MAP_HEIGHT
            )

            frame_interval = 1.0 / self.fps

            while self.video_stream.running():
                current_time = time.time()

                frame, frame_timestamp = self.video_stream.read()

                self.frame_processor.input_queue.put(
                    (frame, lambda f: self.process_frame(f, ball_tracker, mini_map)))

                if not self.frame_processor.output_queue.empty():
                    processed_frame = self.frame_processor.output_queue.get()

                    elapsed = current_time - self.last_frame_time
                    fps = 1.0 / elapsed if elapsed > 0 else 0

                    cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, self.frame_height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow(WINDOW_NAME, processed_frame)

                    processing_time = time.time() - current_time
                    wait_time = max(1, int((frame_interval - processing_time) * 1000))

                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break

                    self.last_frame_time = current_time

            self.cleanup()

    def calculate_fps(self, start_time):
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            return 1.0 / (sum(self.frame_times) / len(self.frame_times))

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