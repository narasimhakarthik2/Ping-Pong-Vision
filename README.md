# 🏓 Ping Pong Vision

A real-time ping pong analysis system powered by computer vision and deep learning. The system tracks ball trajectories, segments the table, detects key points, and provides multiple visualization modes including a stunning glow trail effect.

## 🎯 Demo
![Demo GIF](Data/quad_gif.gif)

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

## 🏗️ System Architecture
```
├── Ping Pong Vision
│   ├── Ball Detection (YOLOv8)
│   ├── Table Analysis
│   │   ├── Semantic Segmentation (U-Net)
│   │   └── Keypoint Detection
│   └── Visualization Pipeline
│       ├── Multi-threaded Processing
│       ├── CUDA Acceleration
│       └── Real-time Rendering
```

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

3. Download pre-trained models
```bash
python download_models.py
```

### Usage
1. Run the main analysis script:
```bash
python main.py --input video.mp4 --output output.mp4
```

2. For real-time webcam analysis:
```bash
python main.py --source 0
```

## 📄 Available Scripts

- `yolo_inference.py`: Main ball detection script
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

## 🔬 Project Structure
```
Ping-Pong-Vision/
├── models/
│   ├── best.pt            # YOLOv8 weights
│   └── table_seg.pth      # U-Net weights
├── utils/
│   ├── ball_tracker.py
│   └── mini_court.py
├── config/
│   └── settings.py
├── SAM/
│   └── model.py
└── requirements.txt
```

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