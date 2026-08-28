import cv2
import numpy as np
import urllib.request
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

BRUSH_SIZE = 8
ERASE_SIZE = 38
INK_COLOR = (255, 255, 255)

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    print("[INFO] Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_detector():
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    return vision.HandLandmarker.create_from_options(options)


def fingers_up(landmarks, handedness_label):
    fingers = []

    if handedness_label == "Right":
        fingers.append(landmarks[TIP_IDS[0]].x < landmarks[PIP_IDS[0]].x)
    else:
        fingers.append(landmarks[TIP_IDS[0]].x > landmarks[PIP_IDS[0]].x)

    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return fingers


def main():
    ensure_model()
    detector = create_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    canvas = None
    prev_point = None
    timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read camera frame")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms += 33
        result = detector.detect_for_video(mp_image, timestamp_ms)

        print("Hands detected:", len(result.hand_landmarks))

        mode = "idle"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            up = fingers_up(landmarks, handedness)
            index_up = up[1]
            middle_up = up[2]
            others_down = not up[3] and not up[4]

            print("Fingers (thumb,index,middle,ring,pinky):", up)

            index_tip = landmarks[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            if index_up and not middle_up and others_down:
                mode = "drawing"
                if prev_point is not None:
                    cv2.line(canvas, prev_point, (x, y), INK_COLOR, BRUSH_SIZE)
                prev_point = (x, y)
                cv2.circle(frame, (x, y), BRUSH_SIZE, (0, 0, 255), 2)

            elif index_up and middle_up and others_down:
                mode = "erasing"
                cv2.circle(canvas, (x, y), ERASE_SIZE, (0, 0, 0), -1)
                cv2.circle(frame, (x, y), ERASE_SIZE, (0, 0, 255), 2)
                prev_point = None

            else:
                prev_point = None
        else:
            prev_point = None

        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        combined = cv2.add(frame_bg, canvas)

        cv2.putText(
            combined, f"mode={mode}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow("Virtual Whiteboard", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()