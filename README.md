# Virtual Whiteboard (Hand Gesture Controlled)

A simple virtual whiteboard built with Python, OpenCV, and MediaPipe Hand Landmarker. Draw and erase on your screen using only hand gestures captured through your webcam — no mouse, no touchscreen, no extra hardware.

## Description

This project uses real-time hand tracking to turn your index finger into a virtual pen. The camera detects 21 landmark points on your hand and interprets specific finger combinations as commands:

- **1 finger (index only) raised** → Drawing mode. Moves a white line following your fingertip.
- **2 fingers (index + middle) raised** → Erasing mode. Clears the canvas in a circular area around your fingertip.

The drawing is layered on top of the live camera feed in real time, so you can see your hand and your drawing at the same time.

## Features

- Real-time hand tracking using MediaPipe Hand Landmarker
- Gesture-based draw/erase switching, no keyboard or mouse needed while drawing
- Adjustable brush size, erase size, and ink color
- Full canvas clear with a single keypress
- Works with any standard webcam

## Requirements

- Python 3.9 or newer
- A working webcam

## Installation

Clone this repository, then install the required libraries:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

1. Run the script:

```bash
python vboard.py
```

2. On first run, the hand landmarker model will be downloaded automatically (only happens once).

3. A window will open showing your webcam feed.

4. Raise your **index finger only** to draw.

5. Raise your **index and middle fingers together** to erase.

6. Controls:
   - Press `c` to clear the entire canvas
   - Press `q` to quit the program

## How It Works

1. OpenCV captures frames from the webcam.
2. Each frame is passed to MediaPipe's Hand Landmarker, which returns 21 (x, y) landmark points per detected hand.
3. For each finger, the fingertip position is compared to the joint below it to determine if the finger is extended or bent.
4. Based on which fingers are extended, the program switches between drawing mode and erasing mode.
5. Drawings are stored on a separate canvas layer and composited over the live camera feed every frame.

## Notes

- Detection accuracy depends on lighting conditions. A well-lit environment improves tracking reliability.
- On Windows, if the camera fails to open or opens slowly, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.

## License

Free to use and modify for personal or educational purposes.
