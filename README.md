# 🏓 Ping Pong Vision

A real-time ping pong analysis system powered by computer vision and deep learning. The system tracks ball trajectories, segments the table, detects key points, and provides multiple visualization modes including a stunning glow trail effect.

## 🎯 Demo
![Demo GIF](Data/quad_gif.gif)
![Demo GIF](Data/inference.gif)
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

### Installation
1. Clone the repository
```bash
git clone https://github.com/narasimhakarthik2/Ping-Pong-Vision.git
cd Ping-Pong-Vision
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Dataset Preparation

### 1. Dataset Structure
```
data/
├── videos/
│   ├── game_1.mp4
│   ├── game_2.mp4
│   └── ...
├── annotations/
│   ├── table/
│   │   ├── game_1.json
│   │   └── ...
│   └── ball/
│       ├── game_1.txt
│       └── ...
└── images/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

### 2. Download and Setup
```bash
# Clone the repository if you haven't already
git clone https://github.com/narasimhakarthik2/Ping-Pong-Vision.git
cd Ping-Pong-Vision

# Navigate to dataset preparation directory
cd prepare_dataset

# Download dataset and unzip annotations
python download_dataset.py
python unzip.py
```

### 3. Extract Images from Videos
For full dataset extraction:
```bash
python extract_all_images.py
```

For custom extraction settings:
```bash
python extract_all_images.py --fps 30 --output_dir path/to/output
```

#### Arguments:
- `--fps`: Frame extraction rate (default: 30)
- `--output_dir`: Output directory for extracted frames
- `--video_dir`: Input video directory (default: data/videos)
- `--start_frame`: Starting frame number (default: 0)
- `--end_frame`: Ending frame number (default: -1 for all frames)

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

## 🙏 Acknowledgments
- YOLOv8 team for the object detection framework
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework

## 📞 Contact
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

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