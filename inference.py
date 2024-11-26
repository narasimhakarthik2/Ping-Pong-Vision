import cv2
import torch
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
from SAM.model import UNet
from utils.mini_court import TableTennisMiniMap
from utils.ball_tracker import BallTracker
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


class PingPongAnalyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_models()
        self.setup_video()
        self.setup_display()
        self.frame_times = deque(maxlen=FPS_BUFFER_SIZE)
        self.cached_mask = None
        self.segmentation_interval = 100000000000000
        self.frame_count = 0
        self.stored_keypoints = None
        self.last_frame_time = 0

        # Initialize video writer
        self.output_path = 'Data/inference_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        # Initialize frame processor
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
            try:
                self.keypoint_model.load_state_dict(
                    torch.load('table_key_points/saved_models/best_model.pth', map_location=self.device)[
                        'model_state_dict']
                )
                self.keypoint_model.to(self.device).eval()
                print(f"Keypoint detection model loaded successfully on {self.device}")
            except Exception as e:
                print(f"Warning: Could not load keypoint model: {e}")
                self.keypoint_model = None

        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

    def detect_keypoints(self, frame):
        """Detect table keypoints in the frame with position adjustment."""
        if self.stored_keypoints is not None:
            return self.stored_keypoints

        if self.keypoint_model is None:
            return None

        with torch.no_grad():
            # Preprocess image
            image = cv2.resize(frame, (224, 224))
            image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            image = image.to(self.device)

            # Get predictions
            keypoints = self.keypoint_model(image)
            keypoints = keypoints.cpu().numpy().reshape(-1, 2)

            # Scale keypoints back to original image size
            scale_x = frame.shape[1] / 224
            scale_y = frame.shape[0] / 224
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

            # Apply offsets
            offset_x = 90
            offset_y = 15
            keypoints[:, 0] += offset_x
            keypoints[:, 1] += offset_y

            # Add extra adjustment to left top point
            left_top_x_offset = -10
            keypoints[0, 0] += left_top_x_offset

            self.stored_keypoints = keypoints
            self.keypoint_model = None  # Unload the model

            return keypoints

    def draw_keypoints(self, frame, keypoints):
        if keypoints is None:
            return frame

        overlay = np.zeros_like(frame)

        # Draw table outline
        corner_indices = [0, 1, 3, 2, 0]  # LEFT_TOP, RIGHT_TOP, RIGHT_BOTTOM, LEFT_BOTTOM, LEFT_TOP
        for i in range(len(corner_indices) - 1):
            pt1 = tuple(map(int, keypoints[corner_indices[i]]))
            pt2 = tuple(map(int, keypoints[corner_indices[i + 1]]))
            cv2.line(overlay, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
        return frame

    def process_frame(self, frame, ball_tracker, mini_map):
        current_time = time.time()

        # Update segmentation mask
        if self.cached_mask is None or self.frame_count % self.segmentation_interval == 0:
            with torch.no_grad():
                input_frame = cv2.resize(frame, (256, 256))
                input_tensor = torch.from_numpy(input_frame).permute(2, 0, 1).unsqueeze(0).float().to(
                    self.device) / 255.0
                segmentation_mask = self.table_segmentation_model(input_tensor)
                segmentation_mask = segmentation_mask.squeeze().cpu().numpy()
                self.cached_mask = cv2.resize(segmentation_mask, (frame.shape[1], frame.shape[0]))

        self.frame_count += 1

        # Get keypoints
        if self.stored_keypoints is None:
            keypoints = self.detect_keypoints(frame)
        else:
            keypoints = self.stored_keypoints

        # Draw keypoints on frame
        if keypoints is not None:
            frame = self.draw_keypoints(frame, keypoints)

        # Apply segmentation mask
        overlay = frame.copy()
        overlay[self.cached_mask > 0.5] = [0, 255, 0]
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.7, 0)

        # Ball tracking
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, device=self.device)
        ball_tracker.update(results[0], frame.shape[1], current_time)
        frame = ball_tracker.draw_trajectory(frame)

        # Update mini-map
        current_pos = ball_tracker.get_current_position()
        if current_pos:
            y_pos = int(current_pos[1])
            x_pos = int(current_pos[0])

            is_impact = False
            if (self.cached_mask is not None and
                    0 <= y_pos < frame.shape[0] and
                    0 <= x_pos < frame.shape[1]):
                is_impact = self.cached_mask[y_pos, x_pos] > 0.5

            frame = mini_map.draw(frame,
                                  ball_pos=current_pos,
                                  keypoints=self.stored_keypoints,
                                  is_impact=is_impact)
        else:
            frame = mini_map.draw(frame)

        return frame

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

            # Read frame with timestamp
            frame, frame_timestamp = self.video_stream.read()

            # Process frame
            self.frame_processor.input_queue.put(
                (frame, lambda f: self.process_frame(f, ball_tracker, mini_map)))

            if not self.frame_processor.output_queue.empty():
                processed_frame = self.frame_processor.output_queue.get()

                # Calculate actual FPS
                elapsed = current_time - self.last_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0

                cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Write frame to output video
                self.video_writer.write(processed_frame)

                cv2.imshow(WINDOW_NAME, processed_frame)

                # Calculate time to wait
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
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        print(f"Output video saved to: {self.output_path}")


def main():
    torch.backends.cudnn.benchmark = True
    try:
        analyzer = PingPongAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()