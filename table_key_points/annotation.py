import cv2
import numpy as np
import json
from pathlib import Path


class PingPongAnnotator:
    def __init__(self):
        self.keypoint_names = [
            "LEFT_TOP",
            "RIGHT_TOP",
            "LEFT_BOTTOM",
            "RIGHT_BOTTOM"
        ]

        self.colors = {
            'table_edges': (0, 255, 0),  # Green
            'points': (0, 0, 255)  # Red
        }

        self.keypoints = []
        self.current_point = 0
        self.image = None
        self.window_name = "Ping Pong Table Annotator"

    def draw_annotations(self):
        img_copy = self.image.copy()

        # Draw points and labels
        for i, (x, y) in enumerate(self.keypoints):
            cv2.circle(img_copy, (x, y), 5, self.colors['points'], -1)
            cv2.putText(img_copy, f"{i + 1}:{self.keypoint_names[i]}", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw table edges as a rectangle
        if len(self.keypoints) >= 4:
            # Draw left vertical line
            cv2.line(img_copy, self.keypoints[0], self.keypoints[2],
                     self.colors['table_edges'], 2)
            # Draw right vertical line
            cv2.line(img_copy, self.keypoints[1], self.keypoints[3],
                     self.colors['table_edges'], 2)
            # Draw top horizontal line
            cv2.line(img_copy, self.keypoints[0], self.keypoints[1],
                     self.colors['table_edges'], 2)
            # Draw bottom horizontal line
            cv2.line(img_copy, self.keypoints[2], self.keypoints[3],
                     self.colors['table_edges'], 2)

        # Show current instruction
        if self.current_point < len(self.keypoint_names):
            cv2.putText(img_copy, f"Mark point: {self.keypoint_names[self.current_point]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(img_copy, "Press 's' to save or 'r' to reset",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(self.window_name, img_copy)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point < len(self.keypoint_names):
                self.keypoints.append((x, y))
                self.current_point += 1
                self.draw_annotations()

    def annotate_image(self, image_path):
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            print(f"Error: Could not load image {image_path}")
            return None

        self.keypoints = []
        self.current_point = 0

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"\nAnnotating: {image_path}")
        print("\nAnnotation Instructions:")
        print("1. Click to mark each keypoint in order:")
        for i, name in enumerate(self.keypoint_names):
            print(f"   {i + 1}: {name}")
        print("\n2. Press 'r' to reset if you make a mistake")
        print("3. Press 's' to save when done")
        print("4. Press 'q' to quit\n")

        while True:
            self.draw_annotations()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # Reset
                self.keypoints = []
                self.current_point = 0
                self.draw_annotations()
            elif key == ord('s') and len(self.keypoints) == len(self.keypoint_names):
                cv2.destroyAllWindows()
                return self.keypoints
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return None

    def save_annotation(self, image_path, annotations_dir):
        if not self.keypoints:
            return None

        keypoints_dict = {
            "image_path": str(image_path),
            "image_size": self.image.shape[:2],
            "keypoints": {
                name: {"x": int(point[0]), "y": int(point[1])}
                for name, point in zip(self.keypoint_names, self.keypoints)
            }
        }

        image_name = Path(image_path).stem
        annotation_path = annotations_dir / f"{image_name}.json"

        with open(annotation_path, 'w') as f:
            json.dump(keypoints_dict, f, indent=4)
        print(f"Saved annotation to {annotation_path}")
        return annotation_path


def annotate_frames(image_dir, annotations_dir):
    # Convert paths to Path objects
    image_dir = Path(image_dir)
    annotations_dir = Path(annotations_dir)

    # Create annotations directory if it doesn't exist
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Get sorted list of image files
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    if not image_paths:
        print("No images found in directory")
        return

    total_images = len(image_paths)
    print(f"\nFound {total_images} images")
    print(f"Annotations will be saved to: {annotations_dir}")

    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n--- Processing image {i}/{total_images} ---")

        # Check if annotation already exists
        annotation_path = annotations_dir / f"{image_path.stem}.json"
        if annotation_path.exists():
            user_input = input(f"Annotation already exists for {image_path.name}. Override? (y/n): ")
            if user_input.lower() != 'y':
                continue

        # Annotate image
        annotator = PingPongAnnotator()
        keypoints = annotator.annotate_image(str(image_path))

        if keypoints is None:
            print("Annotation cancelled")
            user_input = input("Continue to next image? (y/n): ")
            if user_input.lower() != 'y':
                break
            continue

        # Save annotation
        annotator.save_annotation(image_path, annotations_dir)

    print("\nAnnotation process completed!")


if __name__ == "__main__":
    # Define paths
    image_dir = Path("../dataset/training/images/game_1")
    annotations_dir = Path("annotations")

    # Run manual annotation
    annotate_frames(image_dir, annotations_dir)