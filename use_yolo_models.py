"""
Test script to use the trained YOLOv8-pose models directly
Since these are TensorFlow.js models, we'll load them using TensorFlow.js converter
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Try loading TensorFlow and check if model can be loaded
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.version.VERSION}")
    print("TensorFlow loaded successfully!")
except Exception as e:
    print(f"Error loading TensorFlow: {e}")

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Test camera
cap = cv2.VideoCapture(0)

print("\n=== Workout Detector ===")
print("Currently using MediaPipe pose detection")
print("Press 'q' to quit")
print("\n** NOTE: TensorFlow.js model conversion unsuccessful **")
print("The models you uploaded are in TensorFlow.js format (YOLO pose detection).")
print("These models need to be converted to Python TensorFlow format first.")
print("\nRECOMMENDATION:")
print("1. Export your YOLOv8-pose models from Colab as .pt (PyTorch) format")
print("2. Or export as TensorFlow SavedModel format from Python")
print("3. TensorFlow.js models require special handling and have compatibility issues\n")

frame_count = 0
fps_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate FPS
    frame_count += 1
    if (time.time() - fps_time) > 1:
        fps = frame_count / (time.time() - fps_time)
        frame_count = 0
        fps_time = time.time()
    else:
        fps = 0
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.Pose().process(rgb_frame)
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        # Display message
        cv2.putText(frame, "MediaPipe Pose Detection (NO TRAINED MODEL LOADED)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Models need conversion from TensorFlow.js to Python TF", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    if fps > 0:
        cv2.putText(frame, f'FPS: {int(fps)}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Workout Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
