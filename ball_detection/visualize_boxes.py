import cv2
import json
import os
from glob import glob


def visualize_boxes(image_path, ball_data, box_sizes=[20, 22]):
    """
    Visualize different box sizes around the ball center point

    Args:
        image_path: path to the image
        ball_data: dict with 'x' and 'y' coordinates
        box_sizes: list of box sizes to try (in pixels)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    # Create copies for different box sizes
    for size in box_sizes:
        img_copy = img.copy()
        half_size = size // 2

        x, y = ball_data['x'], ball_data['y']

        # Draw box
        cv2.rectangle(img_copy,
                      (x - half_size, y - half_size),
                      (x + half_size, y + half_size),
                      (0, 255, 0), 2)

        # Add size label
        cv2.putText(img_copy, f'Box size: {size}px',
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Show image
        cv2.imshow(f'Box Size {size}', img_copy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    dataset_dir = '../dataset/training'
    game_dirs = glob(os.path.join(dataset_dir, 'images/game_*'))

    for game_dir in game_dirs:
        game_name = os.path.basename(game_dir)
        annotation_file = os.path.join(dataset_dir, 'annotations', game_name, 'ball_markup.json')

        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Get first few frames with ball visible
        valid_frames = [(frame_num, data) for frame_num, data in annotations.items()
                        if data['x'] != -1 and data['y'] != -1][:5]

        for frame_num, ball_data in valid_frames:
            image_path = os.path.join(game_dir, f'img_{frame_num.zfill(6)}.jpg')
            print(f"Visualizing {image_path}")
            visualize_boxes(image_path, ball_data)

            user_input = input("Press Enter to continue to next frame, 'q' to quit: ")
            if user_input.lower() == 'q':
                return


if __name__ == '__main__':
    main()