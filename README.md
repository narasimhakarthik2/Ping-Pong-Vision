# 🏓 Ping Pong Vision

A real-time ping pong analysis system powered by computer vision and deep learning. The system tracks ball trajectories, segments the table, detects key points, and provides multiple visualization modes including a stunning glow trail effect.

## 🎯 Demo
## Quad View Analysis
Multi-view analysis system showing original footage, ball tracking, table segmentation, and combined visualization.

![Demo GIF](Data/quad_gif.gif)

## Inference
Real-time ball detection, table segmentation, and trajectory analysis with top-down tactical view.

![Demo GIF](Data/inference.gif)

## Ball Trajectory Trail Effect
Beautiful visualization of ball movement with dynamic glowing trail effect.

![Demo GIF](Data/trial-effect.gif)

## Key Features
- **Real-time Ball Detection & Tracking**
  - YOLOv8-based ball detection
  - Smooth trajectory tracking
  - Glowing trail effect visualization
  - 20+ FPS performance on consumer GPUs

- **Table Analysis**
  - Semantic segmentation for table detection
  - Keypoint detection for corner identification
  - Top-down view transformation
  - Real-time bounce detection

- **Multi-View Analysis System**
  - Original gameplay footage
  - Ball tracking overlay
  - Table segmentation view
  - Top-down tactical analysis
  - Combined visualization mode

## Architecture
- The U-Net architecture used in this project for table segmentation
![U-Net](Data/unet.png)

## 🛠️ Technical Implementation

### High-Performance Processing
- Multi-threaded architecture with producer-consumer pattern
- Thread-safe queues for frame management
- CUDA-accelerated model inference
- Optimized memory management for real-time processing

### Ball Tracking System
- Custom-trained YOLOv8 model for ball detection
- Trajectory smoothing algorithms
- Dynamic trail effect rendering
- Impact point detection

### Table Analysis
- U-Net architecture for semantic segmentation
- Custom keypoint detection model
- Perspective transformation for top-down view
- Real-time boundary tracking

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
CUDA Toolkit 11.0+
PyTorch 1.9+
OpenCV 4.5+
```

# 🚀 Setup Instructions

## 1. Initial Setup
```bash
# Clone the repository
git clone https://github.com/narasimhakarthik2/Ping-Pong-Vision.git
cd Ping-Pong-Vision

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Dataset Preparation
```bash
# Navigate to dataset preparation directory
cd prepare_dataset

# Download dataset and unzip annotations
python download_dataset.py
python unzip.py

# Extract frames from videos
python extract_all_images.py
```

## 3. Training Pipeline

### A. Ball Detection Model (YOLOv8)
```bash
# Navigate to ball detection directory
cd ball_detection

# Prepare YOLO dataset format
python yolo_dataset.py

# Train YOLOv8 model
python train_yolov8.py
```
This will create a trained model in `models/best.pt`

### B. Table Segmentation Model (SAM)
```bash
# Navigate to SAM directory
cd SAM

# Generate table annotations using SAM
python annotate_table.py

# Train segmentation model
python train_seg.py
```
The trained segmentation model will be saved in `SAM/output/table_segmentation_model.pth`

### C. Table Keypoints Model
```bash
# Navigate to table keypoints directory
cd table_key_points

# Create keypoint annotations
python annotation.py

# Train keypoint detection model
python train.py
```
The keypoint model will be saved in `table_key_points/saved_models/best_model.pth`

### Usage
1. Run the main analysis script:
```bash
python inference.py
```

## 📄 Available Scripts

- `quad_view.py`: Multi-view visualization script
- `ball_trail_effect.py`: Trail effect implementation

## 🎮 Controls
- `q`: Quit the application
- `s`: Save current frame
- `spacebar`: Pause/Resume

## 📊 Performance Metrics
- Average FPS: 20-30 (depending on hardware)
- Detection Accuracy: ~95%
- Latency: <50ms

## 🖥️ System Requirements
### Minimum:
- CUDA-capable GPU (4GB VRAM)
- 8GB RAM
- Intel i5 or equivalent

### Recommended:
- NVIDIA GPU (8GB+ VRAM)
- 16GB RAM
- Intel i7 or equivalent


## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact
- LinkedIn: https://www.linkedin.com/in/narasimha-karthik/
- Email: narasimhakarthik2@gmail.com

## 📚 Citation
If you use this project in your research or work, please cite:
```
@software{ping_pong_vision,
  author = {Your Name},
  title = {Ping Pong Vision},
  year = {2024},
  url = {https://github.com/narasimhakarthik2/Ping-Pong-Vision}
}
```