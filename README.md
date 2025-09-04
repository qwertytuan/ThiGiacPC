# ThiGiacPC - Hand Gesture Recognition System

A powerful real-time hand gesture recognition system that uses computer vision and machine learning to detect hand gestures and execute system commands. Built with MediaPipe, TensorFlow, and OpenCV.

## 🌟 Features

- **Real-time Hand Gesture Recognition**: Detect and classify hand gestures using your webcam
- **Command Execution**: Map hand gestures to system commands (volume control, applications, etc.)
- **Multiple Applications**:
  - Streamlit web interface for easy interaction
  - Standalone OpenCV application for direct usage
  - Data logging tool for training custom gestures
- **Customizable Gestures**: Train your own hand gestures and commands
- **Pre-trained Models**: Comes with models for common gestures (Open, Close, Pointer, OK, Thumbs Up/Down, etc.)
- **Configurable Settings**: Adjust camera settings, detection confidence, and more

## 🎯 Pre-trained Gestures

The system comes with the following pre-trained hand gestures:
- **Open Hand** - Open palm gesture
- **Close Hand** - Closed fist gesture  
- **Pointer** - Index finger pointing
- **OK Sign** - OK hand gesture
- **Thumbs Up** - Positive gesture
- **Thumbs Down** - Negative gesture
- **Turn Left** - Navigation gesture

## 🔧 Requirements

- Python 3.12.7 or higher
- Webcam (built-in or external)
- Operating System: Windows, macOS, or Linux

### Dependencies

The project uses the following main libraries:
- **MediaPipe** - Hand landmark detection
- **TensorFlow/Keras** - Machine learning models
- **OpenCV** - Computer vision processing
- **Streamlit** - Web interface
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/qwertytuan/ThiGiacPC.git
   cd ThiGiacPC
   ```

2. **Install dependencies using uv (recommended)**:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

   Or using pip directly:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import mediapipe, cv2, streamlit, tensorflow; print('All dependencies installed successfully!')"
   ```

## 🚀 Usage

### 1. Streamlit Web Application (Recommended)

The easiest way to use the hand gesture recognition system:

```bash
streamlit run TGPC.py
```

**Features of the web app**:
- Interactive camera controls (device selection, resolution, flip settings)
- Real-time gesture detection and display
- Assign custom commands to gestures
- View existing gesture-command mappings
- Adjustable detection confidence and tracking parameters

### 2. Standalone OpenCV Application

For direct usage without a web interface:

```bash
python .app.py --device 0 --width 1280 --height 720
```

**Command line options**:
- `--device`: Camera device ID (default: 0)
- `--width`: Camera width in pixels (default: 1280)
- `--height`: Camera height in pixels (default: 720)
- `--use_static_image_mode`: Enable static image mode
- `--min_detection_confidence`: Minimum detection confidence (default: 0.7)
- `--min_tracking_confidence`: Minimum tracking confidence (default: 0.5)

**Keyboard Controls**:
- `ESC`: Exit the application
- `k`: Switch to gesture logging mode
- `h`: Switch to point history logging mode
- `n`: Switch to normal mode
- `0-9`: Log gesture data with corresponding number

### 3. Data Logging for Custom Training

To collect data for training your own gestures:

```bash
streamlit run inputdata.py
```

This tool allows you to:
- Capture hand gesture data
- Label your custom gestures
- Log keypoint data or point history data
- Build datasets for training new models

## 📂 Project Structure

```
ThiGiacPC/
│
├── TGPC.py                           # Main Streamlit web application
├── .app.py                           # Standalone OpenCV application  
├── inputdata.py                      # Data logging tool for training
├── hand_sign_commands.csv            # Gesture-to-command mappings
├── requirements.txt                  # Python dependencies
├── pyproject.toml                   # Project configuration
│
├── model/                           # Machine learning models
│   ├── keypoint_classifier/         # Hand pose classification
│   │   ├── keypoint_classifier.py
│   │   ├── keypoint_classifier.keras
│   │   ├── keypoint_classifier.tflite
│   │   └── keypoint_classifier_label.csv
│   │
│   └── point_history_classifier/    # Gesture trajectory classification
│       ├── point_history_classifier.py
│       └── point_history_classifier_label.csv
│
├── utils/                           # Utility functions
│   ├── cvfpscalc.py                # FPS calculation
│   └── __init__.py
│
├── .streamlit/                      # Streamlit configuration
│   └── config.toml
│
├── keypoint_classification_EN.ipynb    # Training notebook for keypoints
└── point_history_classification_EN.ipynb # Training notebook for trajectories
```

## ⚙️ Configuration

### Custom Commands

Edit `hand_sign_commands.csv` to map gestures to system commands:

```csv
Gesture Name,Command
Thumb Up,amixer sset Master 10%+
Thumb Down,amixer sset Master 10%-
OK,echo "Hello World"
```

### Camera Settings

In the Streamlit app, you can adjust:
- Camera device (web cam or external cam)
- Resolution (width/height)
- Image flip settings
- Detection and tracking confidence thresholds
- Maximum number of hands to detect

## 🎓 Training Custom Gestures

1. **Collect Data**:
   ```bash
   streamlit run inputdata.py
   ```
   - Enter a label for your gesture
   - Select mode (Keypoint or Point History)
   - Capture multiple samples of your gesture

2. **Train Models**:
   - Use the provided Jupyter notebooks:
     - `keypoint_classification_EN.ipynb` for hand pose classification
     - `point_history_classification_EN.ipynb` for gesture trajectory classification

3. **Update Labels**:
   - Your new gestures will be automatically added to the label files
   - The system will recognize your custom gestures in real-time

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Try different device IDs (0, 1, 2)
   - Ensure no other applications are using the camera

2. **Poor gesture recognition**:
   - Adjust `min_detection_confidence` and `min_tracking_confidence`
   - Ensure good lighting conditions
   - Keep hand clearly visible to the camera
   - Train with more gesture samples

3. **Commands not executing**:
   - Check `hand_sign_commands.csv` format
   - Ensure commands are valid for your operating system
   - Verify file permissions for command execution

4. **Performance issues**:
   - Lower camera resolution
   - Reduce `max_num_hands` parameter
   - Close other resource-intensive applications

### System Requirements

- **Minimum**: 4GB RAM, dual-core processor, basic webcam
- **Recommended**: 8GB RAM, quad-core processor, HD webcam

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe** team for the excellent hand tracking solution
- **TensorFlow** for the machine learning framework
- **OpenCV** community for computer vision tools
- **Streamlit** for the easy-to-use web framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Look through existing GitHub issues
3. Create a new issue with detailed information about your problem

---

**Happy Gesture Recognition! 👋**