#!/usr/bin/env python3
"""
Workout Buddy - YOLOv8-pose Models (Primary) with MediaPipe Fallback
Uses trained YOLOv8 keypoints for accurate form assessment with confidence scoring
"""

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Base path to models
MODELS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Model mappings
MODEL_PATHS = {
    'pushup': os.path.join(MODELS_BASE, 'WarmApp_PushUp_Model', 'best.pt'),
    'pullup': os.path.join(MODELS_BASE, 'WarmApp_PullUp_Model_V2', 'best.pt'),
    'glute_bridge': os.path.join(MODELS_BASE, 'WarmApp_GluteBridge_Model', 'best.pt'),
    'good_morning': os.path.join(MODELS_BASE, 'WarmApp_GoodMorning_Model', 'best.pt'),
    'plank': os.path.join(MODELS_BASE, 'WarmApp_Plank_Medium_Model', 'best.pt'),
}

# YOLO COCO Keypoint indices (17 keypoints)
YOLO_KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def calculate_angle_with_confidence(a, b, c):
    """
    Calculates angle and returns weighted confidence based on keypoint confidences.
    Returns: (angle, confidence_score)
    """
    angle = calculate_angle(a[:2], b[:2], c[:2])
    # Weight by minimum confidence of the three points
    confidence = min(a[2], b[2], c[2]) if len(a) > 2 else 1.0
    return angle, confidence

def yolo_keypoints_to_coords(keypoints, img_shape):
    """
    Convert YOLOv8 keypoint format to pixel coordinates.
    YOLOv8 returns keypoints as [x, y, confidence] for each of 17 keypoints.
    
    Returns: list of (x, y, conf) tuples
    """
    h, w = img_shape[:2]
    coords = []
    for i in range(0, len(keypoints), 3):
        x = int(keypoints[i] * w)
        y = int(keypoints[i+1] * h)
        conf = keypoints[i+2]
        coords.append((x, y, conf))
    return coords

class YOLOKeypointMapper:
    """Maps YOLO keypoints to exercise-specific indices and extracts angles."""
    
    def __init__(self, keypoints_coords):
        """
        Args:
            keypoints_coords: list of (x, y, conf) tuples from YOLO (17 keypoints)
        """
        self.kp = keypoints_coords
        
    def get_point(self, yolo_idx):
        """Get point by YOLO index with confidence."""
        if 0 <= yolo_idx < len(self.kp):
            return self.kp[yolo_idx]
        return None
    
    def is_visible(self, yolo_idx, threshold=0.5):
        """Check if keypoint is visible above confidence threshold."""
        pt = self.get_point(yolo_idx)
        return pt is not None and pt[2] > threshold
    
    def get_angle_with_conf(self, idx_a, idx_b, idx_c):
        """Calculate angle between three YOLO keypoints with confidence."""
        pt_a = self.get_point(idx_a)
        pt_b = self.get_point(idx_b)
        pt_c = self.get_point(idx_c)
        
        if pt_a and pt_b and pt_c:
            return calculate_angle_with_confidence(pt_a, pt_b, pt_c)
        return None, 0.0

def calculate_form_confidence_weighted(angle_data, ideal_ranges):
    """
    Calculate form quality with confidence weighting from YOLO keypoints.
    
    Args:
        angle_data: list of (angle, confidence) tuples
        ideal_ranges: list of (min, max) tuples for ideal angles
    
    Returns: (classification, overall_confidence, form_score)
    """
    total_score = 0
    total_weight = 0
    
    for (angle, conf), (min_val, max_val) in zip(angle_data, ideal_ranges):
        # Calculate how well angle matches ideal range
        if min_val <= angle <= max_val:
            angle_score = 1.0
        else:
            if angle < min_val:
                deviation = min_val - angle
            else:
                deviation = angle - max_val
            angle_score = max(0.0, 1.0 - (deviation / 30.0))
        
        # Weight by keypoint confidence
        weighted_score = angle_score * conf
        total_score += weighted_score
        total_weight += conf
    
    # Overall form score (0-1)
    form_score = total_score / total_weight if total_weight > 0 else 0.0
    avg_confidence = total_weight / len(angle_data) if angle_data else 0.0
    
    # Classification based on form score
    if form_score >= 0.85:
        classification = "Excellent"
    elif form_score >= 0.70:
        classification = "Good"
    elif form_score >= 0.50:
        classification = "Fair"
    else:
        classification = "Poor"
    
    return classification, avg_confidence, form_score

# =============================================================================
# EXERCISE CONFIGURATIONS (YOLO Keypoint Indices)
# =============================================================================

# Push-up keypoints (YOLO indices: shoulder, hip, knee / shoulder, elbow, wrist)
pushup_configs = {
    'hips': [(YOLO_KP['left_shoulder'], YOLO_KP['left_hip'], YOLO_KP['left_knee']),
             (YOLO_KP['right_shoulder'], YOLO_KP['right_hip'], YOLO_KP['right_knee'])],
    'arms': [(YOLO_KP['right_shoulder'], YOLO_KP['right_elbow'], YOLO_KP['right_wrist']),
             (YOLO_KP['left_shoulder'], YOLO_KP['left_elbow'], YOLO_KP['left_wrist'])],
    'ideal_hip': (160, 200),
    'ideal_arm_down': (80, 110),
    'ideal_arm_up': (160, 180)
}

# Glute Bridge
glute_bridge_configs = {
    'hips': [(YOLO_KP['left_shoulder'], YOLO_KP['left_hip'], YOLO_KP['left_knee']),
             (YOLO_KP['right_shoulder'], YOLO_KP['right_hip'], YOLO_KP['right_knee'])],
    'knees': [(YOLO_KP['left_hip'], YOLO_KP['left_knee'], YOLO_KP['left_ankle']),
              (YOLO_KP['right_hip'], YOLO_KP['right_knee'], YOLO_KP['right_ankle'])],
    'ideal_hip': (160, 200),
    'ideal_knee': (80, 110)
}

# Good Morning
good_morning_configs = {
    'back': [(YOLO_KP['left_shoulder'], YOLO_KP['left_hip'], YOLO_KP['left_knee']),
             (YOLO_KP['right_shoulder'], YOLO_KP['right_hip'], YOLO_KP['right_knee'])],
    'knees': [(YOLO_KP['left_hip'], YOLO_KP['left_knee'], YOLO_KP['left_ankle']),
              (YOLO_KP['right_hip'], YOLO_KP['right_knee'], YOLO_KP['right_ankle'])],
    'ideal_back_down': (45, 90),
    'ideal_back_up': (150, 180),
    'ideal_knee': (160, 200)
}

# Plank
plank_configs = {
    'back': [(YOLO_KP['left_shoulder'], YOLO_KP['left_hip'], YOLO_KP['left_knee']),
             (YOLO_KP['right_shoulder'], YOLO_KP['right_hip'], YOLO_KP['right_knee'])],
    'shoulders': [(YOLO_KP['left_elbow'], YOLO_KP['left_shoulder'], YOLO_KP['left_hip']),
                  (YOLO_KP['right_elbow'], YOLO_KP['right_shoulder'], YOLO_KP['right_hip'])],
    'ideal_back': (160, 200),
    'ideal_shoulder': (80, 110)
}

# Pull-up
pullup_configs = {
    'arms': [(YOLO_KP['right_shoulder'], YOLO_KP['right_elbow'], YOLO_KP['right_wrist']),
             (YOLO_KP['left_shoulder'], YOLO_KP['left_elbow'], YOLO_KP['left_wrist'])],
    'shoulders': [(YOLO_KP['right_elbow'], YOLO_KP['right_shoulder'], YOLO_KP['right_hip']),
                  (YOLO_KP['left_elbow'], YOLO_KP['left_shoulder'], YOLO_KP['left_hip'])],
    'ideal_arm_up': (60, 90),
    'ideal_arm_down': (160, 180)
}

# =============================================================================
# MAIN PROGRAM
# =============================================================================

print("=" * 60)
print("  Workout Buddy - AI-Powered with YOLOv8 Exercise Tracking")
print("=" * 60)
print("\nSelect exercise to track:")
print("1. Push-ups")
print("2. Glute Bridge")
print("3. Good Morning")
print("4. Plank")
print("5. Pull-ups")
exercise_choice = input("Enter choice (1-5): ")

exercise_map = {
    '1': 'pushup',
    '2': 'glute_bridge',
    '3': 'good_morning',
    '4': 'plank',
    '5': 'pullup'
}

selected_exercise = exercise_map.get(exercise_choice, 'pushup')
print(f"\nTracking: {selected_exercise.replace('_', ' ').title()}")

# Load YOLOv8 model for selected exercise
model_path = MODEL_PATHS.get(selected_exercise)
yolo_model = None
yolo_available = False

if model_path and os.path.exists(model_path):
    try:
        print(f"Loading YOLOv8 model: {model_path}")
        yolo_model = YOLO(model_path)
        yolo_available = True
        print("✓ YOLOv8 Model Loaded Successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLOv8 model: {e}")
        print("  Falling back to MediaPipe only")
else:
    print(f"✗ Model not found at: {model_path}")
    print("  Using MediaPipe only")

print("✓ MediaPipe Pose Detection Active")
print("\nPress ESC to exit\n")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Metrics
rep_count = 0
stage = "up" if selected_exercise != 'plank' else "inactive"
metrics = {
    'yolo_confidence': 0.0,
    'yolo_detection': False,
    'ai_classification': 'N/A',
    'ai_confidence': 0.0,
    'form_quality': 'N/A',
    'hip_angle': 0,
    'arm_angle': 0,
    'knee_angle': 0,
    'back_angle': 0
}

frame_counter = 0
update_interval = 5

print("Starting camera feed...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # =============================================================================
    # YOLO INFERENCE (if available)
    # =============================================================================
    yolo_keypoints = None
    if yolo_available:
        try:
            results = yolo_model(frame_rgb, verbose=False)
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data
                if len(keypoints_data) > 0:
                    # Get first detected person
                    kp = keypoints_data[0].cpu().numpy()
                    yolo_keypoints = kp.flatten()
                    
                    # Get detection confidence
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        metrics['yolo_confidence'] = float(results[0].boxes[0].conf[0])
                        metrics['yolo_detection'] = True
                    
                    # Draw YOLO keypoints (17 keypoints from YOLOv8-pose)
                    coords = yolo_keypoints_to_coords(yolo_keypoints, frame.shape)
                    for idx, (x, y, conf) in enumerate(coords):
                        if conf > 0.5:  # Only draw confident keypoints
                            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                            cv2.circle(frame, (x, y), 4, color, -1)
                            # cv2.putText(frame, f"{idx}", (x+5, y-5), 
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    metrics['yolo_detection'] = False
        except Exception as e:
            print(f"YOLO inference error: {e}")
            metrics['yolo_detection'] = False
    
    # =============================================================================
    # MEDIAPIPE INFERENCE
    # =============================================================================
    mp_results = pose.process(frame_rgb)
    
    if mp_results.pose_landmarks:
        landmarks = mp_results.pose_landmarks.landmark
        
        # Draw MediaPipe skeleton (lighter colors to not conflict with YOLO)
        mp_drawing.draw_landmarks(
            frame, 
            mp_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200,150,100), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(200,100,180), thickness=1, circle_radius=1)
        )
        
        # ==================== PUSH-UP DETECTION ====================
        if selected_exercise == 'pushup':
            hip_angles = []
            arm_angles = []
            
            for hips in pushup_hips:
                points = [landmarks[i] for i in hips]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                hip_angles.append(angle)
            
            for arms in pushup_arms:
                points = [landmarks[i] for i in arms]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                arm_angles.append(angle)
            
            metrics['hip_angle'] = int(np.mean(hip_angles))
            metrics['arm_angle'] = int(np.mean(arm_angles))
            
            # Form scoring
            if frame_counter % update_interval == 0:
                avg_arm_angle = np.mean(arm_angles)
                if avg_arm_angle < 130:
                    ideal_ranges = [pushup_ideal_hip, pushup_ideal_arm_down]
                    angles = [metrics['hip_angle'], metrics['arm_angle']]
                else:
                    ideal_ranges = [pushup_ideal_hip, pushup_ideal_arm_up]
                    angles = [metrics['hip_angle'], metrics['arm_angle']]
                
                classification, confidence = calculate_form_confidence_simple(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            # Rep counting
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle > 160 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_arm_angle < 100:
                stage = "down"
        
        # ==================== GLUTE BRIDGE ====================
        elif selected_exercise == 'glute_bridge':
            hip_angles = []
            knee_angles = []
            
            for hips in glute_bridge_hips:
                points = [landmarks[i] for i in hips]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                hip_angles.append(angle)
            
            for knees in glute_bridge_knees:
                points = [landmarks[i] for i in knees]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                knee_angles.append(angle)
            
            metrics['hip_angle'] = int(np.mean(hip_angles))
            metrics['knee_angle'] = int(np.mean(knee_angles))
            
            if frame_counter % update_interval == 0:
                ideal_ranges = [glute_bridge_ideal_hip, glute_bridge_ideal_knee]
                angles = [metrics['hip_angle'], metrics['knee_angle']]
                classification, confidence = calculate_form_confidence_simple(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            avg_hip_angle = np.mean(hip_angles)
            if avg_hip_angle > 160 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_hip_angle < 120:
                stage = "down"
        
        # ==================== GOOD MORNING ====================
        elif selected_exercise == 'good_morning':
            back_angles = []
            knee_angles = []
            
            for back in good_morning_back:
                points = [landmarks[i] for i in back]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                back_angles.append(angle)
            
            for knees in good_morning_knees:
                points = [landmarks[i] for i in knees]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                knee_angles.append(angle)
            
            metrics['back_angle'] = int(np.mean(back_angles))
            metrics['knee_angle'] = int(np.mean(knee_angles))
            
            if frame_counter % update_interval == 0:
                avg_back_angle = np.mean(back_angles)
                if avg_back_angle < 120:
                    ideal_ranges = [good_morning_ideal_back_down, good_morning_ideal_knee]
                else:
                    ideal_ranges = [good_morning_ideal_back_up, good_morning_ideal_knee]
                angles = [metrics['back_angle'], metrics['knee_angle']]
                classification, confidence = calculate_form_confidence_simple(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            avg_back_angle = np.mean(back_angles)
            if avg_back_angle > 150 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_back_angle < 90:
                stage = "down"
        
        # ==================== PLANK ====================
        elif selected_exercise == 'plank':
            back_angles = []
            shoulder_angles = []
            
            for back in plank_back:
                points = [landmarks[i] for i in back]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                back_angles.append(angle)
            
            for shoulders in plank_shoulders:
                points = [landmarks[i] for i in shoulders]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                shoulder_angles.append(angle)
            
            metrics['back_angle'] = int(np.mean(back_angles))
            metrics['arm_angle'] = int(np.mean(shoulder_angles))
            
            if frame_counter % update_interval == 0:
                ideal_ranges = [plank_ideal_back, plank_ideal_shoulder]
                angles = [metrics['back_angle'], metrics['arm_angle']]
                classification, confidence = calculate_form_confidence_simple(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            avg_back_angle = np.mean(back_angles)
            if plank_ideal_back[0] <= avg_back_angle <= plank_ideal_back[1]:
                if stage == "inactive":
                    stage = "holding"
                    rep_count += 1
            else:
                stage = "inactive"
        
        # ==================== PULL-UP ====================
        elif selected_exercise == 'pullup':
            arm_angles = []
            shoulder_angles = []
            
            for arms in pullup_arms:
                points = [landmarks[i] for i in arms]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                arm_angles.append(angle)
            
            for shoulders in pullup_shoulders:
                points = [landmarks[i] for i in shoulders]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                shoulder_angles.append(angle)
            
            metrics['arm_angle'] = int(np.mean(arm_angles))
            
            if frame_counter % update_interval == 0:
                avg_arm_angle = np.mean(arm_angles)
                if avg_arm_angle < 120:
                    ideal_ranges = [pullup_ideal_arm_up]
                else:
                    ideal_ranges = [pullup_ideal_arm_down]
                angles = [metrics['arm_angle']]
                classification, confidence = calculate_form_confidence_simple(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle < 90 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_arm_angle > 160:
                stage = "down"
        
        frame_counter += 1
    
    # =============================================================================
    # DISPLAY UI
    # =============================================================================
    
    # Header
    cv2.putText(frame, f"Exercise: {selected_exercise.replace('_', ' ').title()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Stage: {stage}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # YOLOv8 Model Status
    y_offset = 120
    if yolo_available:
        status_color = (0, 255, 0) if metrics['yolo_detection'] else (0, 165, 255)
        cv2.putText(frame, "=== YOLOv8 Model ===", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        cv2.putText(frame, f"Detection: {'YES' if metrics['yolo_detection'] else 'NO'}", 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        if metrics['yolo_detection']:
            cv2.putText(frame, f"Confidence: {metrics['yolo_confidence']:.1%}", 
                        (10, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 80
    else:
        cv2.putText(frame, "YOLOv8: Not Loaded", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += 30
    
    # Form Analysis
    cv2.putText(frame, "=== Form Analysis ===", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Quality: {metrics['ai_classification']}", 
                (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Score: {metrics['ai_confidence']:.1%}", 
                (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Angle metrics
    y_offset += 90
    cv2.putText(frame, "=== Angles ===", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if metrics['hip_angle'] > 0:
        cv2.putText(frame, f"Hip: {metrics['hip_angle']}°", 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if metrics['arm_angle'] > 0:
        cv2.putText(frame, f"Arm: {metrics['arm_angle']}°", 
                    (10, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if metrics['knee_angle'] > 0:
        cv2.putText(frame, f"Knee: {metrics['knee_angle']}°", 
                    (10, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if metrics['back_angle'] > 0:
        cv2.putText(frame, f"Back: {metrics['back_angle']}°", 
                    (10, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Workout Buddy - YOLOv8 + MediaPipe', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 60}")
print("  Workout Summary")
print(f"{'=' * 60}")
print(f"Exercise: {selected_exercise.replace('_', ' ').title()}")
print(f"Total Reps: {rep_count}")
print(f"Final Form Quality: {metrics['ai_classification']}")
print(f"Final Confidence: {metrics['ai_confidence']:.1%}")
if yolo_available:
    print(f"YOLOv8 Detection Rate: {'Active' if metrics['yolo_detection'] else 'N/A'}")
print("\nKeep up the great work!")
