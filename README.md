Real-Time Face Mask Detection System

A real-time computer vision application that detects faces and classifies mask usage (Mask / No Mask) using OpenCV and a TensorFlow/Keras deep learning model (mask_detector.h5).

📌 Features

Real-time detection from live camera/webcam feed.

Facial recognition via OpenCV Haar Cascade classifier.

TensorFlow/Keras deep learning model inference.

Colored bounding box overlays and confidence rating indicators.

🛠️ Requirements

Python Version: Python 3.10 – 3.12 (Python 3.14+ is not supported due to OpenCV binary compatibility issues)

OS: Windows, macOS, or Linux

Dependencies: tensorflow, opencv-python, numpy, pillow

🚀 Quick Start Guide

1. Clone the Repository
```bash
git clone https://github.com/Tanmay8k/mask-detector
cd mask-detector
```
NOTE: if python 3.12 is not installed:
```bash
py install 3.12
```
2. Set Up Virtual Environment (venv)

Create a fresh virtual environment using Python 3.12:

On Windows (PowerShell):
```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

Note: If you receive a script execution error in PowerShell, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


On Windows (Command Prompt):
```bash
py -3.12 -m venv venv
.\venv\Scripts\activate.bat
```

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

Upgrade pip and install all necessary packages inside the activated (venv) environment:
```bash
python -m pip install --upgrade pip
pip install tensorflow "opencv-python<5.0.0" numpy pillow
```

4. System Verification Commands

Verify that your environment, packages, and Haar Cascades are correctly initialized before running the project:

# Check OpenCV, TensorFlow, and NumPy versions
```bash
python -c "import cv2, tensorflow as tf, numpy as np; print(f'OpenCV: {cv2.__version__}\nTensorFlow: {tf.__version__}\nNumPy: {np.__version__}')"
```
# Test Haar Cascade classifier loading
```bash
python -c "import cv2; cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'); print('Cascade Status:', 'Success' if not cascade.empty() else 'Failed')"
```

5. Run the Application

Launch the real-time detector:
```bash
python detect_mask.py
```

Controls:

ESC or q: Close the camera feed and safely exit the application.

📁 Repository Structure

Face-Mask-Detection/
│
├── detect_mask.py                     # Main execution script
├── mask_detector.h5                   # Pre-trained TensorFlow model weights
├── haarcascade_frontalface_default.xml# Haar Cascade face detection XML file
├── requirements.txt                   # Dependency requirements file
├── README.md                          # Project documentation
└── venv/                              # Virtual environment (ignored in git)


❓ Troubleshooting

1. AttributeError: module 'cv2' has no attribute 'CascadeClassifier'

This indicates an incompatible OpenCV version (e.g., experimental OpenCV 5.0). Fix it by reinstalling OpenCV 4.x:

pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install --no-cache-dir "opencv-python<5.0.0"


2. TypeError: ord() expected a character, but string of length 3 found

Ensure your exit key check in detect_mask.py uses integer keycodes rather than multi-character strings:

# Use 27 for ESC or ord('q') for 'q'
if cv2.waitKey(1) & 0xFF == 27:
    break


🤝 Contributing & License

Feel free to fork this repository, submit issues, and make pull requests!
