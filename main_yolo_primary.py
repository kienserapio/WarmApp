#!/usr/bin/env python3
"""
Workout Buddy - YOLOv8 Primary Detection
Uses trained YOLOv8 keypoints for form assessment with confidence-weighted scoring
MediaPipe serves as fallback only
"""

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

MODEL_PATHS = {
    'pushup': os.path.join(MODELS_BASE, 'WarmApp_PushUp_Model', 'best.pt'),
    'pullup': os.path.join(MODELS_BASE, 'WarmApp_PullUp_Model_V2', 'best.pt'),
    'glute_bridge': os.path.join(MODELS_BASE, 'WarmApp_GluteBridge_Model', 'best.pt'),
    'good_morning': os.path.join(MODELS_BASE, 'WarmApp_GoodMorning_Model', 'best.pt'),
    'plank': os.path.join(MODELS_BASE, 'WarmApp_Plank_Medium_Model', 'best.pt'),
}

# YOLO COCO Keypoints (17 keypoints)
KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees."""
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def draw_yolo_skeleton(frame, kp_mapper, conf_threshold=0.5):
    """
    Draw full YOLO skeleton like MediaPipe style.
    Shows all keypoints and connections with confidence-based colors.
    """
    # YOLO skeleton connections (similar to MediaPipe POSE_CONNECTIONS)
    connections = [
        # Face
        (KP['nose'], KP['left_eye']), (KP['left_eye'], KP['left_ear']),
        (KP['nose'], KP['right_eye']), (KP['right_eye'], KP['right_ear']),
        # Upper body
        (KP['left_shoulder'], KP['right_shoulder']),
        (KP['left_shoulder'], KP['left_elbow']), (KP['left_elbow'], KP['left_wrist']),
        (KP['right_shoulder'], KP['right_elbow']), (KP['right_elbow'], KP['right_wrist']),
        (KP['left_shoulder'], KP['left_hip']), (KP['right_shoulder'], KP['right_hip']),
        # Lower body
        (KP['left_hip'], KP['right_hip']),
        (KP['left_hip'], KP['left_knee']), (KP['left_knee'], KP['left_ankle']),
        (KP['right_hip'], KP['right_knee']), (KP['right_knee'], KP['right_ankle']),
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_pt = kp_mapper.get_point(start_idx)
        end_pt = kp_mapper.get_point(end_idx)
        
        if start_pt and end_pt and start_pt[2] > conf_threshold and end_pt[2] > conf_threshold:
            # Color based on confidence
            avg_conf = (start_pt[2] + end_pt[2]) / 2
            if avg_conf > 0.7:
                color = (245, 117, 66)  # Blue (high confidence)
            elif avg_conf > 0.5:
                color = (245, 180, 66)  # Cyan (medium confidence)
            else:
                color = (180, 180, 180)  # Gray (low confidence)
            
            cv2.line(frame, tuple(map(int, start_pt[:2])), tuple(map(int, end_pt[:2])), 
                    color, 2, cv2.LINE_AA)
    
    # Draw keypoints
    for idx in range(17):
        pt = kp_mapper.get_point(idx)
        if pt and pt[2] > conf_threshold:
            # Color based on confidence
            if pt[2] > 0.7:
                color = (245, 66, 230)  # Pink (high confidence)
            elif pt[2] > 0.5:
                color = (245, 180, 66)  # Cyan (medium confidence)
            else:
                color = (180, 180, 180)  # Gray (low confidence)
            
            cv2.circle(frame, tuple(map(int, pt[:2])), 4, color, -1)
            cv2.circle(frame, tuple(map(int, pt[:2])), 5, (255, 255, 255), 1)

def draw_angle_visualization(frame, pts, angle, ideal_range, confidence, label=""):
    """
    Draw angle visualization with color coding on top of skeleton:
    - GREEN: Good form (angle within ideal range)
    - RED: Poor form (angle outside range)
    - YELLOW: Warning (borderline)
    
    Thickness varies by confidence.
    """
    min_val, max_val = ideal_range
    
    # Determine color based on form quality
    if min_val <= angle <= max_val:
        color = (0, 255, 0)  # GREEN - Good form
        status = "✓"
    elif abs(angle - min_val) < 10 or abs(angle - max_val) < 10:
        color = (0, 255, 255)  # YELLOW - Warning
        status = "!"
    else:
        color = (0, 0, 255)  # RED - Poor form
        status = "✗"
    
    # Line thickness based on confidence (3-5px for visibility over skeleton)
    thickness = int(3 + confidence * 2)
    
    # Draw thicker lines for angle measurement
    cv2.line(frame, tuple(map(int, pts[0][:2])), tuple(map(int, pts[1][:2])), color, thickness)
    cv2.line(frame, tuple(map(int, pts[1][:2])), tuple(map(int, pts[2][:2])), color, thickness)
    
    # Draw larger keypoint circles for angle points
    for pt in pts:
        cv2.circle(frame, tuple(map(int, pt[:2])), 6, color, -1)
        cv2.circle(frame, tuple(map(int, pt[:2])), 7, (255, 255, 255), 2)
    
    # Draw angle text at middle point
    mid_pt = tuple(map(int, pts[1][:2]))
    text = f"{status} {int(angle)}°"
    if label:
        text = f"{label}: {text}"
    
    # Text background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (mid_pt[0]-5, mid_pt[1]-text_h-8), 
                  (mid_pt[0]+text_w+5, mid_pt[1]+5), (0, 0, 0), -1)
    cv2.putText(frame, text, (mid_pt[0], mid_pt[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def calculate_form_score(angle_data, ideal_ranges):
    """
    Calculate weighted form score from YOLO keypoint confidences.
    
    Args:
        angle_data: [(angle, confidence), ...]
        ideal_ranges: [(min, max), ...]
    
    Returns: (classification, avg_conf, form_score, keypoint_quality)
    """
    total_score = 0
    total_conf = 0
    
    for (angle, conf), (min_val, max_val) in zip(angle_data, ideal_ranges):
        # Angle accuracy score
        if min_val <= angle <= max_val:
            angle_score = 1.0
        else:
            deviation = min(abs(angle - min_val), abs(angle - max_val))
            angle_score = max(0.0, 1.0 - (deviation / 30.0))
        
        # Weight by keypoint confidence
        total_score += angle_score * conf
        total_conf += conf
    
    form_score = total_score / total_conf if total_conf > 0 else 0.0
    avg_conf = total_conf / len(angle_data) if angle_data else 0.0
    
    # Classification
    if form_score >= 0.85:
        classification = "Excellent"
    elif form_score >= 0.70:
        classification = "Good"
    elif form_score >= 0.50:
        classification = "Fair"
    else:
        classification = "Poor"
    
    # Keypoint quality assessment
    if avg_conf >= 0.8:
        kp_quality = "High"
    elif avg_conf >= 0.6:
        kp_quality = "Medium"
    else:
        kp_quality = "Low"
    
    return classification, avg_conf, form_score, kp_quality

# =============================================================================
# EXERCISE-SPECIFIC PROCESSING
# =============================================================================

def process_pushup(kp_mapper, frame):
    """Process push-up exercise using YOLO keypoints."""
    hip_angles = []
    arm_angles = []
    
    # Hip alignment (shoulder-hip-knee)
    for side in ['left', 'right']:
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_shoulder'], KP[f'{side}_hip'], KP[f'{side}_knee'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_hip']),
                   kp_mapper.get_point(KP[f'{side}_knee'])]
            draw_angle_visualization(frame, pts, angle, (160, 200), conf, "Hip")
            hip_angles.append((angle, conf))
    
    # Arm bend (shoulder-elbow-wrist)
    for side in ['left', 'right']:
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_shoulder'], KP[f'{side}_elbow'], KP[f'{side}_wrist'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_elbow']),
                   kp_mapper.get_point(KP[f'{side}_wrist'])]
            # Determine ideal range based on position
            ideal_range = (80, 110) if angle < 130 else (160, 180)
            draw_angle_visualization(frame, pts, angle, ideal_range, conf, "Arm")
            arm_angles.append((angle, conf))
    
    return hip_angles, arm_angles

def process_glute_bridge(kp_mapper, frame):
    """Process glute bridge exercise."""
    hip_angles = []
    knee_angles = []
    
    for side in ['left', 'right']:
        # Hip extension
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_shoulder'], KP[f'{side}_hip'], KP[f'{side}_knee'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_hip']),
                   kp_mapper.get_point(KP[f'{side}_knee'])]
            draw_angle_visualization(frame, pts, angle, (160, 200), conf, "Hip")
            hip_angles.append((angle, conf))
        
        # Knee angle
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_hip'], KP[f'{side}_knee'], KP[f'{side}_ankle'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_hip']),
                   kp_mapper.get_point(KP[f'{side}_knee']),
                   kp_mapper.get_point(KP[f'{side}_ankle'])]
            draw_angle_visualization(frame, pts, angle, (80, 110), conf, "Knee")
            knee_angles.append((angle, conf))
    
    return hip_angles, knee_angles

def process_good_morning(kp_mapper, frame):
    """Process good morning exercise."""
    back_angles = []
    knee_angles = []
    
    for side in ['left', 'right']:
        # Back angle (hip hinge)
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_shoulder'], KP[f'{side}_hip'], KP[f'{side}_knee'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_hip']),
                   kp_mapper.get_point(KP[f'{side}_knee'])]
            ideal_range = (45, 90) if angle < 120 else (150, 180)
            draw_angle_visualization(frame, pts, angle, ideal_range, conf, "Back")
            back_angles.append((angle, conf))
        
        # Knee (should stay straight)
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_hip'], KP[f'{side}_knee'], KP[f'{side}_ankle'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_hip']),
                   kp_mapper.get_point(KP[f'{side}_knee']),
                   kp_mapper.get_point(KP[f'{side}_ankle'])]
            draw_angle_visualization(frame, pts, angle, (160, 200), conf, "Knee")
            knee_angles.append((angle, conf))
    
    return back_angles, knee_angles

def process_plank(kp_mapper, frame):
    """Process plank exercise."""
    back_angles = []
    shoulder_angles = []
    
    for side in ['left', 'right']:
        # Back alignment
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_shoulder'], KP[f'{side}_hip'], KP[f'{side}_knee'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_hip']),
                   kp_mapper.get_point(KP[f'{side}_knee'])]
            draw_angle_visualization(frame, pts, angle, (160, 200), conf, "Back")
            back_angles.append((angle, conf))
        
        # Shoulder position
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_elbow'], KP[f'{side}_shoulder'], KP[f'{side}_hip'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_elbow']),
                   kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_hip'])]
            draw_angle_visualization(frame, pts, angle, (80, 110), conf, "Shoulder")
            shoulder_angles.append((angle, conf))
    
    return back_angles, shoulder_angles

def process_pullup(kp_mapper, frame):
    """Process pull-up exercise."""
    arm_angles = []
    shoulder_angles = []
    
    for side in ['left', 'right']:
        # Arm bend
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_shoulder'], KP[f'{side}_elbow'], KP[f'{side}_wrist'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_elbow']),
                   kp_mapper.get_point(KP[f'{side}_wrist'])]
            ideal_range = (60, 90) if angle < 120 else (160, 180)
            draw_angle_visualization(frame, pts, angle, ideal_range, conf, "Arm")
            arm_angles.append((angle, conf))
        
        # Shoulder engagement
        angle, conf = kp_mapper.get_angle_with_conf(
            KP[f'{side}_elbow'], KP[f'{side}_shoulder'], KP[f'{side}_hip'])
        if angle is not None:
            pts = [kp_mapper.get_point(KP[f'{side}_elbow']),
                   kp_mapper.get_point(KP[f'{side}_shoulder']),
                   kp_mapper.get_point(KP[f'{side}_hip'])]
            shoulder_angles.append((angle, conf))
    
    return arm_angles, shoulder_angles

# =============================================================================
# YOLO KEYPOINT MAPPER CLASS
# =============================================================================

class YOLOKeypointMapper:
    """Extract and map YOLO keypoints for angle calculations."""
    
    def __init__(self, keypoints_raw, img_shape):
        """
        Args:
            keypoints_raw: flattened array from YOLO [x1,y1,conf1, x2,y2,conf2, ...]
            img_shape: (height, width) for pixel conversion
        """
        h, w = img_shape[:2]
        self.keypoints = []
        for i in range(0, len(keypoints_raw), 3):
            x = int(keypoints_raw[i] * w)
            y = int(keypoints_raw[i+1] * h)
            conf = float(keypoints_raw[i+2])
            self.keypoints.append((x, y, conf))
    
    def get_point(self, idx):
        """Get keypoint by YOLO index."""
        if 0 <= idx < len(self.keypoints):
            return self.keypoints[idx]
        return None
    
    def is_visible(self, idx, threshold=0.5):
        """Check if keypoint is visible."""
        pt = self.get_point(idx)
        return pt is not None and pt[2] > threshold
    
    def get_angle_with_conf(self, idx_a, idx_b, idx_c):
        """Calculate angle with confidence weighting."""
        pts = [self.get_point(i) for i in [idx_a, idx_b, idx_c]]
        if all(pts) and all(p[2] > 0.3 for p in pts):  # Min confidence threshold
            angle = calculate_angle(pts[0], pts[1], pts[2])
            confidence = min(p[2] for p in pts)  # Use minimum confidence
            return angle, confidence
        return None, 0.0

# =============================================================================
# MAIN PROGRAM
# =============================================================================

print("=" * 70)
print("  Workout Buddy - YOLOv8 Primary Keypoint Detection")
print("=" * 70)
print("\nSelect exercise:")
print("1. Push-ups")
print("2. Glute Bridge")
print("3. Good Morning")
print("4. Plank")
print("5. Pull-ups")
choice = input("Enter (1-5): ")

exercise_map = {'1': 'pushup', '2': 'glute_bridge', '3': 'good_morning', 
                '4': 'plank', '5': 'pullup'}
selected_exercise = exercise_map.get(choice, 'pushup')

print(f"\nTracking: {selected_exercise.replace('_', ' ').title()}")

# Load YOLO model
model_path = MODEL_PATHS.get(selected_exercise)
if not model_path or not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

print(f"Loading YOLOv8 model...")
yolo_model = YOLO(model_path)
print("✓ YOLOv8 Loaded")
print("✓ Using YOLO keypoints for all measurements")
print("\nPress ESC to exit\n")

# Initialize MediaPipe (fallback only)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# State tracking
rep_count = 0
stage = "up" if selected_exercise != 'plank' else "inactive"
metrics = {
    'detection_conf': 0.0,
    'form_class': 'N/A',
    'form_score': 0.0,
    'keypoint_quality': 'N/A',
    'avg_angle_1': 0,
    'avg_angle_2': 0,
}
frame_counter = 0
update_interval = 3

print("Camera starting...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ==========================================================================
    # YOLO INFERENCE
    # ==========================================================================
    yolo_results = yolo_model(frame_rgb, verbose=False)
    kp_mapper = None
    yolo_detected = False
    
    if len(yolo_results) > 0 and yolo_results[0].keypoints is not None:
        kp_data = yolo_results[0].keypoints.data
        if len(kp_data) > 0:
            kp_flat = kp_data[0].cpu().numpy().flatten()
            kp_mapper = YOLOKeypointMapper(kp_flat, frame.shape)
            yolo_detected = True
            
            # Detection confidence
            if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
                metrics['detection_conf'] = float(yolo_results[0].boxes[0].conf[0])
    
    # ==========================================================================
    # DRAW SKELETON (like MediaPipe)
    # ==========================================================================
    if kp_mapper and yolo_detected:
        # Draw full skeleton first (like MediaPipe)
        draw_yolo_skeleton(frame, kp_mapper, conf_threshold=0.5)
    
    # ==========================================================================
    # EXERCISE PROCESSING (YOLO KEYPOINTS)
    # ==========================================================================
    if kp_mapper and yolo_detected:
        angle_data_1 = []
        angle_data_2 = []
        
        if selected_exercise == 'pushup':
            hip_angles, arm_angles = process_pushup(kp_mapper, frame)
            angle_data_1, angle_data_2 = hip_angles, arm_angles
            
            # Rep counting
            if arm_angles:
                avg_arm = np.mean([a[0] for a in arm_angles])
                metrics['avg_angle_1'] = int(np.mean([a[0] for a in hip_angles])) if hip_angles else 0
                metrics['avg_angle_2'] = int(avg_arm)
                if avg_arm > 160 and stage == "down":
                    stage = "up"
                    rep_count += 1
                elif avg_arm < 100:
                    stage = "down"
        
        elif selected_exercise == 'glute_bridge':
            hip_angles, knee_angles = process_glute_bridge(kp_mapper, frame)
            angle_data_1, angle_data_2 = hip_angles, knee_angles
            
            if hip_angles:
                avg_hip = np.mean([a[0] for a in hip_angles])
                metrics['avg_angle_1'] = int(avg_hip)
                metrics['avg_angle_2'] = int(np.mean([a[0] for a in knee_angles])) if knee_angles else 0
                if avg_hip > 160 and stage == "down":
                    stage = "up"
                    rep_count += 1
                elif avg_hip < 120:
                    stage = "down"
        
        elif selected_exercise == 'good_morning':
            back_angles, knee_angles = process_good_morning(kp_mapper, frame)
            angle_data_1, angle_data_2 = back_angles, knee_angles
            
            if back_angles:
                avg_back = np.mean([a[0] for a in back_angles])
                metrics['avg_angle_1'] = int(avg_back)
                metrics['avg_angle_2'] = int(np.mean([a[0] for a in knee_angles])) if knee_angles else 0
                if avg_back > 150 and stage == "down":
                    stage = "up"
                    rep_count += 1
                elif avg_back < 90:
                    stage = "down"
        
        elif selected_exercise == 'plank':
            back_angles, shoulder_angles = process_plank(kp_mapper, frame)
            angle_data_1, angle_data_2 = back_angles, shoulder_angles
            
            if back_angles:
                avg_back = np.mean([a[0] for a in back_angles])
                metrics['avg_angle_1'] = int(avg_back)
                metrics['avg_angle_2'] = int(np.mean([a[0] for a in shoulder_angles])) if shoulder_angles else 0
                if 160 <= avg_back <= 200:
                    if stage == "inactive":
                        stage = "holding"
                        rep_count += 1
                else:
                    stage = "inactive"
        
        elif selected_exercise == 'pullup':
            arm_angles, shoulder_angles = process_pullup(kp_mapper, frame)
            angle_data_1, angle_data_2 = arm_angles, shoulder_angles
            
            if arm_angles:
                avg_arm = np.mean([a[0] for a in arm_angles])
                metrics['avg_angle_1'] = int(avg_arm)
                metrics['avg_angle_2'] = int(np.mean([a[0] for a in shoulder_angles])) if shoulder_angles else 0
                if avg_arm < 90 and stage == "down":
                    stage = "up"
                    rep_count += 1
                elif avg_arm > 160:
                    stage = "down"
        
        # Form scoring (every N frames)
        if frame_counter % update_interval == 0 and (angle_data_1 or angle_data_2):
            all_angles = angle_data_1 + angle_data_2
            if selected_exercise == 'pushup':
                ideal_ranges = [(160, 200)] * len(angle_data_1) + [(80, 110) if metrics['avg_angle_2'] < 130 else (160, 180)] * len(angle_data_2)
            elif selected_exercise == 'glute_bridge':
                ideal_ranges = [(160, 200)] * len(angle_data_1) + [(80, 110)] * len(angle_data_2)
            elif selected_exercise == 'good_morning':
                ideal_ranges = [(45, 90) if metrics['avg_angle_1'] < 120 else (150, 180)] * len(angle_data_1) + [(160, 200)] * len(angle_data_2)
            elif selected_exercise == 'plank':
                ideal_ranges = [(160, 200)] * len(angle_data_1) + [(80, 110)] * len(angle_data_2)
            elif selected_exercise == 'pullup':
                ideal_ranges = [(60, 90) if metrics['avg_angle_1'] < 120 else (160, 180)] * len(angle_data_1)
            
            if all_angles and ideal_ranges:
                classification, avg_conf, form_score, kp_quality = calculate_form_score(all_angles, ideal_ranges)
                metrics['form_class'] = classification
                metrics['form_score'] = form_score
                metrics['keypoint_quality'] = kp_quality
    
    frame_counter += 1
    
    # ==========================================================================
    # UI DISPLAY
    # ==========================================================================
    
    # Header bar
    cv2.rectangle(frame, (0, 0), (w, 140), (40, 40, 40), -1)
    cv2.putText(frame, f"Exercise: {selected_exercise.replace('_', ' ').title()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}  |  Stage: {stage}", 
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # YOLOv8 Status
    status_color = (0, 255, 0) if yolo_detected else (0, 0, 255)
    cv2.putText(frame, f"YOLOv8: {'ACTIVE' if yolo_detected else 'NO DETECTION'}", 
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    if yolo_detected:
        cv2.putText(frame, f"Detection Conf: {metrics['detection_conf']:.1%}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Form Analysis Panel (right side)
    panel_x = w - 320
    cv2.rectangle(frame, (panel_x, 0), (w, 260), (40, 40, 40), -1)
    
    y = 30
    cv2.putText(frame, "=== FORM ANALYSIS ===", (panel_x + 10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
    y += 35
    
    # Form quality
    form_color = (0, 255, 0) if metrics['form_class'] in ['Excellent', 'Good'] else (0, 165, 255) if metrics['form_class'] == 'Fair' else (0, 0, 255)
    cv2.putText(frame, f"Quality: {metrics['form_class']}", (panel_x + 10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, form_color, 2)
    y += 30
    cv2.putText(frame, f"Form Score: {metrics['form_score']:.1%}", (panel_x + 10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 30
    cv2.putText(frame, f"Keypoint Quality: {metrics['keypoint_quality']}", (panel_x + 10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    y += 40
    cv2.putText(frame, "=== METRICS ===", (panel_x + 10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
    y += 30
    
    # Exercise-specific metrics
    if selected_exercise == 'pushup':
        cv2.putText(frame, f"Hip Angle: {metrics['avg_angle_1']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Arm Angle: {metrics['avg_angle_2']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif selected_exercise == 'glute_bridge':
        cv2.putText(frame, f"Hip Extension: {metrics['avg_angle_1']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Knee Angle: {metrics['avg_angle_2']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif selected_exercise == 'good_morning':
        cv2.putText(frame, f"Back Angle: {metrics['avg_angle_1']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Knee Angle: {metrics['avg_angle_2']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif selected_exercise == 'plank':
        cv2.putText(frame, f"Back Alignment: {metrics['avg_angle_1']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Shoulder Pos: {metrics['avg_angle_2']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif selected_exercise == 'pullup':
        cv2.putText(frame, f"Arm Angle: {metrics['avg_angle_1']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Shoulder Angle: {metrics['avg_angle_2']}°", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Legend at bottom
    legend_y = h - 60
    cv2.rectangle(frame, (0, legend_y), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, "Legend:", (10, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.circle(frame, (90, legend_y + 15), 4, (0, 255, 0), -1)
    cv2.putText(frame, "Good Form", (100, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.circle(frame, (210, legend_y + 15), 4, (0, 255, 255), -1)
    cv2.putText(frame, "Warning", (220, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.circle(frame, (320, legend_y + 15), 4, (0, 0, 255), -1)
    cv2.putText(frame, "Poor Form", (330, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.putText(frame, "Line Thickness = Keypoint Confidence  |  ESC to quit", 
                (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    cv2.imshow('Workout Buddy - YOLOv8 Keypoint Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 70}")
print("  WORKOUT SUMMARY")
print(f"{'=' * 70}")
print(f"Exercise: {selected_exercise.replace('_', ' ').title()}")
print(f"Total Reps: {rep_count}")
print(f"Final Form Quality: {metrics['form_class']}")
print(f"Final Form Score: {metrics['form_score']:.1%}")
print(f"Keypoint Quality: {metrics['keypoint_quality']}")
print("\nGreat work! Keep it up! 💪")
