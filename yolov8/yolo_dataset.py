import os
import json
import shutil
from glob import glob
import random
from pathlib import Path


def create_yolo_label(ball_data, img_width, img_height, box_size=25):
    """Convert ball coordinates to YOLO format"""
    if ball_data['x'] == -1 or ball_data['y'] == -1:
        return None

    # Calculate normalized coordinates
    x_center = ball_data['x'] / img_width
    y_center = ball_data['y'] / img_height

    # Normalize box size
    width = box_size / img_width
    height = box_size / img_height

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def prepare_yolo_dataset(dataset_dir, output_dir, train_ratio=0.8):
    # Create directory structure
    yolo_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for dir_path in yolo_dirs:
        Path(os.path.join(output_dir, dir_path)).mkdir(parents=True, exist_ok=True)

    positive_frames = []  # Frames with ball

    # Process each game
    for game_id in [1, 2, 3]:
        game_name = f'game_{game_id}'
        images_dir = os.path.join(dataset_dir, 'images', game_name)
        annotation_file = os.path.join(dataset_dir, 'annotations', game_name, 'ball_markup.json')

        print(f"\nProcessing {game_name}...")

        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Get all images with ball annotations
        for frame_num, ball_data in annotations.items():
            if ball_data['x'] != -1 and ball_data['y'] != -1:
                img_path = os.path.join(images_dir, f'img_{frame_num.zfill(6)}.jpg')
                if os.path.exists(img_path):
                    positive_frames.append((img_path, frame_num, ball_data))

    print(f"\nDataset statistics:")
    print(f"Total frames with ball: {len(positive_frames)}")

    # Shuffle and split data
    random.shuffle(positive_frames)
    split_idx = int(len(positive_frames) * train_ratio)
    train_frames = positive_frames[:split_idx]
    val_frames = positive_frames[split_idx:]

    # Process train and val sets
    for is_train, frames in [(True, train_frames), (False, val_frames)]:
        subset = 'train' if is_train else 'val'
        for img_path, frame_num, ball_data in frames:
            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            height, width = img.shape[:2]

            # Create YOLO label
            yolo_label = create_yolo_label(ball_data, width, height)
            if yolo_label is None:
                continue

            # Copy image and create label
            new_img_path = os.path.join(output_dir, subset, 'images', os.path.basename(img_path))
            label_path = os.path.join(output_dir, subset, 'labels',
                                      os.path.basename(img_path).replace('.jpg', '.txt'))

            shutil.copy2(img_path, new_img_path)
            with open(label_path, 'w') as f:
                f.write(yolo_label)

    # Create dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

names:
  0: ball
    """

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)


if __name__ == '__main__':
    import cv2

    dataset_dir = '../dataset/training'
    output_dir = '../dataset/yolo_dataset'

    prepare_yolo_dataset(dataset_dir, output_dir)

    # Print summary
    train_images = len(glob(os.path.join(output_dir, 'train/images/*.jpg')))
    val_images = len(glob(os.path.join(output_dir, 'val/images/*.jpg')))
    print(f"\nDataset prepared successfully:")
    print(f"Training images: {train_images}")
    print(f"Validation images: {val_images}")