# Real-Time Hand Gesture Recognition (PyTorch CNN) ✋

A deep learning project implementing a custom Convolutional Neural Network (CNN) in PyTorch to classify hand gestures in real-time using a webcam.

## 📌 Architecture & Training
- **Model**: Custom CNN (`auto_movement`) with 2 Convolutional layers, BatchNorm, Dropout, and 4 Fully Connected layers.
- **Input Resolution**: $64 \times 64$ RGB images.
- **Dataset Classes**: `['F', 'L', 'O', 'W']`.
- **Validation Accuracy**: ~99.38% on test split.

## 🎮 Classes & Actions
| Label | Action | Description |
| :---: | :---: | :---: |
| **F** | FORWARD | OK sign with 3 fingers raised |
| **L** | LEFT | Right-angle gesture (Thumb & Index) |
| **O** | STOP | Circular O gesture |
| **W** | RIGHT | Three vertical fingers |

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision opencv-python pillow numpy
   ```
2. Run live inference:
   ```bash
   python "final project1.py"
   ```
