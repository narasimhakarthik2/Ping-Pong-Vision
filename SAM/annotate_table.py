import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import json
import torch


class TableAnnotator:
    def __init__(self, target_width=960, target_height=540):
        checkpoint_path = "sam_vit_h_4b8939.pth"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

        self.points = []
        self.current_mask = None
        self.scale_factor = None
        self.target_size = (target_width, target_height)
        self.display_frame = None
        print("SAM initialized successfully")

    def draw_instructions(self, frame):
        instructions = [
            "Click points around TABLE TOP ONLY (4+)",
            "'g': generate mask",
            "'r': reset points",
            "'s': save and next",
            "'q': quit"
        ]

        overlay = frame.copy()
        # Draw semi-transparent black background for text
        cv2.rectangle(overlay, (5, 5), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y = 30
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 20
        return frame

    def scale_point(self, x, y, reverse=False):
        if reverse:
            return int(x / self.scale_factor[0]), int(y / self.scale_factor[1])
        else:
            return int(x * self.scale_factor[0]), int(y * self.scale_factor[1])

    def draw_points_and_lines(self):
        frame_copy = self.display_frame.copy()
        frame_copy = self.draw_instructions(frame_copy)

        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                pt1 = self.scale_point(self.points[i][0], self.points[i][1])
                pt2 = self.scale_point(self.points[i + 1][0], self.points[i + 1][1])
                cv2.line(frame_copy, pt1, pt2, (0, 255, 255), 2)

            # Connect back to first point if we have at least 3 points
            if len(self.points) >= 3:
                pt1 = self.scale_point(self.points[-1][0], self.points[-1][1])
                pt2 = self.scale_point(self.points[0][0], self.points[0][1])
                cv2.line(frame_copy, pt1, pt2, (0, 255, 255), 2)

        # Draw points
        for i, point in enumerate(self.points):
            disp_x, disp_y = self.scale_point(point[0], point[1])
            cv2.circle(frame_copy, (disp_x, disp_y), 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(i + 1), (disp_x + 5, disp_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Annotation', frame_copy)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = self.scale_point(x, y, reverse=True)
            self.points.append([orig_x, orig_y])
            self.draw_points_and_lines()

    def generate_mask(self, frame):
        if len(self.points) < 4:
            print("Need at least 4 points")
            return None

        self.predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_points = np.array(self.points)
        input_labels = np.ones(len(self.points))
        masks, _, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        return masks[0]

    def annotate_frame(self, frame):
        self.points = []
        self.current_mask = None

        h, w = frame.shape[:2]
        self.scale_factor = (self.target_size[0] / w, self.target_size[1] / h)
        self.display_frame = cv2.resize(frame, self.target_size)

        window_name = 'Annotation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.target_size[0], self.target_size[1])
        cv2.setMouseCallback(window_name, self.mouse_callback)

        frame_display = self.display_frame.copy()
        frame_display = self.draw_instructions(frame_display)
        cv2.imshow(window_name, frame_display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('g'):
                self.current_mask = self.generate_mask(frame)
                if self.current_mask is not None:
                    # Apply mask overlay to the display frame
                    frame_display = cv2.resize(frame, self.target_size)
                    mask_display = cv2.resize(self.current_mask.astype(np.uint8), self.target_size)
                    frame_display[mask_display == 1] = frame_display[mask_display == 1] * 0.7 + np.array(
                        [0, 255, 0]) * 0.3
                    frame_display = self.draw_instructions(frame_display)
                    self.draw_points_and_lines()  # Keep points visible
                    cv2.imshow(window_name, frame_display)

            elif key == ord('r'):
                self.points = []
                frame_display = self.display_frame.copy()
                frame_display = self.draw_instructions(frame_display)
                cv2.imshow(window_name, frame_display)

            elif key == ord('s') and self.current_mask is not None:
                return self.current_mask

            elif key == ord('q'):
                break

        return None


def process_video(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotator = TableAnnotator()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 30  # Annotate every 30th frame
    frame_count = 0
    annotations = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            print(f"\nFrame {frame_count}/{total_frames}")
            mask = annotator.annotate_frame(frame)

            if mask is not None:
                mask_path = output_dir / f"frame_{frame_count:06d}_mask.npy"
                np.save(mask_path, mask)
                frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                annotations[frame_count] = {
                    'frame': str(frame_path),
                    'mask': str(mask_path)
                }
                print(f"Saved frame {frame_count}")

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    with open(output_dir / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    print("\nAnnotation completed!")


if __name__ == "__main__":
    video_path = "../Data/input.mp4"
    output_dir = "dataset/table_segmentation"
    process_video(video_path, output_dir)