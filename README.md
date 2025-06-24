# Emotion Detector Project 🎭

This project implements a real-time facial emotion detection system using a Convolutional Neural Network (CNN) trained on grayscale 48x48 facial images. The system is capable of detecting the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## 📁 Project Structure

EMOTION-DETECTOR-PROJECT/
├── dataset/
│ ├── train/
│ └── test/
│ ├── angry/
│ ├── disgust/
│ ├── fear/
│ ├── happy/
│ ├── neutral/
│ ├── sad/
│ └── surprise/
├── model/
│ └── emotion_model.h5
├── utils/
│ └── init.py
├── train.py
├── realtime.py
└── README.md

---

## ⚙️ Setup Instructions

1. **Install Dependencies**  
   Ensure you have Python ≥ 3.7 and install the required packages:

   ```bash
   pip install tensorflow opencv-python nump

Dependencies:
TensorFlow

OpenCV (cv2)

NumPy

Dataset Structure:

Source: Kaggle 
Place the training and testing images inside dataset/train/ and dataset/test/.

Each class should have its own folder (angry, happy, etc.), following the format expected by ImageDataGenerator.


Model Architecture:
The CNN is a shallow custom architecture:

Conv2D → ReLU → MaxPooling

Conv2D → ReLU → MaxPooling

Flatten → Dense → Dropout → Output

Output layer uses softmax activation to predict 1 of 7 emotion classes.



TO TRAIN:
python train.py

TO RUN:
python realtime.py


Author
Sourabh Hegde
📍 India
🧠 Real-time Emotion Detection using CNNs
