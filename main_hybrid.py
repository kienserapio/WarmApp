#!/usr/bin/env python3
"""
Workout Buddy - Hybrid MediaPipe + YOLOv8
- MediaPipe: Primary keypoint detection and visualization (works great with partial body)
- YOLOv8: Secondary assessment from your trained models (provides confidence metrics)
- Shows both evaluations side-by-side for comparison
"""

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os

# =============================================================================
# YOLO MODEL CONFIGURATION
# =============================================================================

MODELS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

MODEL_PATHS = {
    'pushup': os.path.join(MODELS_BASE, 'WarmApp_PushUp_Model', 'best.pt'),
    'pullup': os.path.join(MODELS_BASE, 'WarmApp_PullUp_Model_V2', 'best.pt'),
    'glute_bridge': os.path.join(MODELS_BASE, 'WarmApp_GluteBridge_Model', 'best.pt'),
    'good_morning': os.path.join(MODELS_BASE, 'WarmApp_GoodMorning_Model', 'best.pt'),
    'plank': os.path.join(MODELS_BASE, 'WarmApp_Plank_Medium_Model', 'best.pt'),
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

def calculate_form_confidence(angles, ideal_ranges):
    """
    Calculate form quality confidence score based on angle measurements.
    Returns confidence score (0-1) and classification.
    """
    total_score = 0
    count = 0
    
    for angle, (min_val, max_val) in zip(angles, ideal_ranges):
        if min_val <= angle <= max_val:
            score = 1.0
        else:
            if angle < min_val:
                deviation = min_val - angle
            else:
                deviation = angle - max_val
            score = max(0.0, 1.0 - (deviation / 30.0))
        
        total_score += score
        count += 1
    
    confidence = total_score / count if count > 0 else 0.0
    
    if confidence >= 0.8:
        return "Excellent", confidence
    elif confidence >= 0.6:
        return "Good", confidence
    elif confidence >= 0.4:
        return "Fair", confidence
    else:
        return "Poor", confidence

def get_yolo_assessment(yolo_results):
    """
    Extract assessment metrics from YOLO detection.
    Returns: (detected, detection_conf, avg_keypoint_conf, keypoint_quality)
    """
    if len(yolo_results) == 0 or yolo_results[0].keypoints is None:
        return False, 0.0, 0.0, "N/A"
    
    kp_data = yolo_results[0].keypoints.data
    if len(kp_data) == 0:
        return False, 0.0, 0.0, "N/A"
    
    # Detection confidence
    detection_conf = 0.0
    if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
        detection_conf = float(yolo_results[0].boxes[0].conf[0])
    
    # Average keypoint confidence
    kp_flat = kp_data[0].cpu().numpy().flatten()
    confidences = [kp_flat[i] for i in range(2, len(kp_flat), 3)]  # Every 3rd value starting from index 2
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    # Keypoint quality classification
    if avg_conf >= 0.8:
        quality = "High"
    elif avg_conf >= 0.6:
        quality = "Medium"
    else:
        quality = "Low"
    
    return True, detection_conf, avg_conf, quality

# =============================================================================
# MAIN PROGRAM
# =============================================================================

print("=" * 70)
print("  Workout Buddy - Hybrid MediaPipe + YOLOv8 Assessment")
print("=" * 70)
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
print("✓ MediaPipe Pose Detection (Primary)")

# Load YOLO model
model_path = MODEL_PATHS.get(selected_exercise)
yolo_model = None
yolo_available = False

if model_path and os.path.exists(model_path):
    try:
        print("Loading YOLOv8 model for secondary assessment...")
        yolo_model = YOLO(model_path)
        yolo_available = True
        print("✓ YOLOv8 Model Loaded (Secondary Assessment)")
    except Exception as e:
        print(f"✗ Failed to load YOLOv8: {e}")
        print("  Continuing with MediaPipe only")
else:
    print(f"✗ YOLOv8 model not found")
    print("  Continuing with MediaPipe only")

print("\nPress ESC to exit\n")

# Define keypoint triplets and ideal ranges for each exercise
# Push-up keypoints and ideal ranges
pushup_hips = [(11, 23, 25), (12, 24, 26)]  # shoulder-hip-knee
pushup_arms = [(12, 14, 16), (11, 13, 15)]  # shoulder-elbow-wrist
pushup_ideal_hip = (160, 200)
pushup_ideal_arm_down = (80, 110)
pushup_ideal_arm_up = (160, 180)

# Glute Bridge
glute_bridge_hips = [(11, 23, 25), (12, 24, 26)]
glute_bridge_knees = [(23, 25, 27), (24, 26, 28)]
glute_bridge_ideal_hip = (160, 200)
glute_bridge_ideal_knee = (80, 110)

# Good Morning
good_morning_back = [(11, 23, 25), (12, 24, 26)]
good_morning_knees = [(23, 25, 27), (24, 26, 28)]
good_morning_ideal_back_down = (45, 90)
good_morning_ideal_back_up = (150, 180)
good_morning_ideal_knee = (160, 200)

# Plank
plank_back = [(11, 23, 25), (12, 24, 26)]
plank_shoulders = [(13, 11, 23), (14, 12, 24)]
plank_ideal_back = (160, 200)
plank_ideal_shoulder = (80, 110)

# Pull-up
pullup_arms = [(12, 14, 16), (11, 13, 15)]
pullup_shoulders = [(14, 12, 24), (13, 11, 23)]
pullup_ideal_arm_up = (60, 90)
pullup_ideal_arm_down = (160, 180)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

import time

cap = cv2.VideoCapture(0)

# Counters and metrics
rep_count = 0
stage = "up" if selected_exercise != 'plank' else "inactive"
metrics = {
    # MediaPipe metrics
    'mp_classification': 'N/A',
    'mp_confidence': 0.0,
    'hip_angle': 0,
    'arm_angle': 0,
    'knee_angle': 0,
    'back_angle': 0,
    # YOLO metrics
    'yolo_detected': False,
    'yolo_detection_conf': 0.0,
    'yolo_keypoint_conf': 0.0,
    'yolo_keypoint_quality': 'N/A'
}

# Posture alert system
bad_form_timer = 0.0  # Timer in seconds for bad form
bad_form_active = False  # Whether alert is currently active
last_time = None  # For calculating elapsed time

frame_counter = 0
update_interval = 5

print("Camera starting...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Time tracking for alert system
    current_time = time.time()
    if last_time is None:
        last_time = current_time
    delta_time = current_time - last_time
    last_time = current_time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # ==========================================================================
    # YOLO INFERENCE (Secondary Assessment)
    # ==========================================================================
    if yolo_available and frame_counter % 2 == 0:  # Run YOLO every other frame for performance
        try:
            yolo_results = yolo_model(frame_rgb, verbose=False)
            yolo_detected, det_conf, kp_conf, kp_quality = get_yolo_assessment(yolo_results)
            metrics['yolo_detected'] = yolo_detected
            metrics['yolo_detection_conf'] = det_conf
            metrics['yolo_keypoint_conf'] = kp_conf
            metrics['yolo_keypoint_quality'] = kp_quality
        except Exception as e:
            metrics['yolo_detected'] = False
    
    # ==========================================================================
    # MEDIAPIPE INFERENCE (Primary Detection)
    # ==========================================================================
    mp_results = pose.process(frame_rgb)
    
    if mp_results.pose_landmarks:
        landmarks = mp_results.pose_landmarks.landmark
        
        # Draw MediaPipe skeleton
        mp_drawing.draw_landmarks(
            frame, 
            mp_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
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
                
                color = (0, 255, 0) if pushup_ideal_hip[0] <= angle <= pushup_ideal_hip[1] else (0, 0, 255)
                cv2.line(frame, coords[0], coords[1], color, 2)
                cv2.line(frame, coords[1], coords[2], color, 2)
                for coord in coords:
                    cv2.circle(frame, coord, 4, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            for arms in pushup_arms:
                points = [landmarks[i] for i in arms]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                arm_angles.append(angle)
                
                color = (0, 255, 0) if pushup_ideal_arm_down[0] <= angle <= pushup_ideal_arm_down[1] or pushup_ideal_arm_up[0] <= angle <= pushup_ideal_arm_up[1] else (0, 255, 255)
                cv2.line(frame, coords[0], coords[1], color, 2)
                cv2.line(frame, coords[1], coords[2], color, 2)
                for coord in coords:
                    cv2.circle(frame, coord, 4, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            metrics['hip_angle'] = int(np.mean(hip_angles))
            metrics['arm_angle'] = int(np.mean(arm_angles))
            
            if frame_counter % update_interval == 0:
                avg_arm_angle = np.mean(arm_angles)
                if avg_arm_angle < 130:
                    ideal_ranges = [pushup_ideal_hip, pushup_ideal_arm_down]
                    angles = [metrics['hip_angle'], metrics['arm_angle']]
                else:
                    ideal_ranges = [pushup_ideal_hip, pushup_ideal_arm_up]
                    angles = [metrics['hip_angle'], metrics['arm_angle']]
                
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['mp_classification'] = classification
                metrics['mp_confidence'] = confidence
            
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle > 160 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_arm_angle < 100:
                stage = "down"
        
        # ==================== GLUTE BRIDGE DETECTION ====================
        elif selected_exercise == 'glute_bridge':
            hip_angles = []
            knee_angles = []
            
            for hips in glute_bridge_hips:
                points = [landmarks[i] for i in hips]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                hip_angles.append(angle)
                
                color = (0, 255, 0) if glute_bridge_ideal_hip[0] <= angle <= glute_bridge_ideal_hip[1] else (0, 0, 255)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for knees in glute_bridge_knees:
                points = [landmarks[i] for i in knees]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                knee_angles.append(angle)
                
                color = (0, 255, 0) if glute_bridge_ideal_knee[0] <= angle <= glute_bridge_ideal_knee[1] else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['hip_angle'] = int(np.mean(hip_angles))
            metrics['knee_angle'] = int(np.mean(knee_angles))
            
            if frame_counter % update_interval == 0:
                ideal_ranges = [glute_bridge_ideal_hip, glute_bridge_ideal_knee]
                angles = [metrics['hip_angle'], metrics['knee_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['mp_classification'] = classification
                metrics['mp_confidence'] = confidence
            
            avg_hip_angle = np.mean(hip_angles)
            if avg_hip_angle > 160 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_hip_angle < 120:
                stage = "down"
        
        # ==================== GOOD MORNING DETECTION ====================
        elif selected_exercise == 'good_morning':
            back_angles = []
            knee_angles = []
            
            for back in good_morning_back:
                points = [landmarks[i] for i in back]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                back_angles.append(angle)
                
                color = (0, 255, 0) if good_morning_ideal_back_down[0] <= angle <= good_morning_ideal_back_down[1] or good_morning_ideal_back_up[0] <= angle <= good_morning_ideal_back_up[1] else (0, 0, 255)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for knees in good_morning_knees:
                points = [landmarks[i] for i in knees]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                knee_angles.append(angle)
                
                color = (0, 255, 0) if good_morning_ideal_knee[0] <= angle <= good_morning_ideal_knee[1] else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['back_angle'] = int(np.mean(back_angles))
            metrics['knee_angle'] = int(np.mean(knee_angles))
            
            if frame_counter % update_interval == 0:
                avg_back_angle = np.mean(back_angles)
                if avg_back_angle < 120:
                    ideal_ranges = [good_morning_ideal_back_down, good_morning_ideal_knee]
                else:
                    ideal_ranges = [good_morning_ideal_back_up, good_morning_ideal_knee]
                angles = [metrics['back_angle'], metrics['knee_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['mp_classification'] = classification
                metrics['mp_confidence'] = confidence
            
            avg_back_angle = np.mean(back_angles)
            if avg_back_angle > 150 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_back_angle < 90:
                stage = "down"
        
        # ==================== PLANK DETECTION ====================
        elif selected_exercise == 'plank':
            back_angles = []
            shoulder_angles = []
            
            for back in plank_back:
                points = [landmarks[i] for i in back]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                back_angles.append(angle)
                
                color = (0, 255, 0) if plank_ideal_back[0] <= angle <= plank_ideal_back[1] else (0, 0, 255)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for shoulders in plank_shoulders:
                points = [landmarks[i] for i in shoulders]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                shoulder_angles.append(angle)
                
                color = (0, 255, 0) if plank_ideal_shoulder[0] <= angle <= plank_ideal_shoulder[1] else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['back_angle'] = int(np.mean(back_angles))
            metrics['arm_angle'] = int(np.mean(shoulder_angles))
            
            if frame_counter % update_interval == 0:
                ideal_ranges = [plank_ideal_back, plank_ideal_shoulder]
                angles = [metrics['back_angle'], metrics['arm_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['mp_classification'] = classification
                metrics['mp_confidence'] = confidence
            
            avg_back_angle = np.mean(back_angles)
            if plank_ideal_back[0] <= avg_back_angle <= plank_ideal_back[1]:
                if stage == "inactive":
                    stage = "holding"
                    rep_count += 1
            else:
                stage = "inactive"
        
        # ==================== PULL-UP DETECTION ====================
        elif selected_exercise == 'pullup':
            arm_angles = []
            shoulder_angles = []
            
            for arms in pullup_arms:
                points = [landmarks[i] for i in arms]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                arm_angles.append(angle)
                
                color = (0, 255, 0) if pullup_ideal_arm_up[0] <= angle <= pullup_ideal_arm_up[1] or pullup_ideal_arm_down[0] <= angle <= pullup_ideal_arm_down[1] else (0, 255, 255)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for shoulders in pullup_shoulders:
                points = [landmarks[i] for i in shoulders]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                shoulder_angles.append(angle)
                
                color = (0, 255, 0) if 30 <= angle <= 90 else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['arm_angle'] = int(np.mean(arm_angles))
            
            if frame_counter % update_interval == 0:
                avg_arm_angle = np.mean(arm_angles)
                if avg_arm_angle < 120:
                    ideal_ranges = [pullup_ideal_arm_up]
                else:
                    ideal_ranges = [pullup_ideal_arm_down]
                angles = [metrics['arm_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['mp_classification'] = classification
                metrics['mp_confidence'] = confidence
            
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle < 90 and stage == "down":
                stage = "up"
            elif avg_arm_angle > 160:
                stage = "down"

            frame_counter += 1

    # ==========================================================================
    if rep_count > 0 and metrics['mp_classification'] != 'N/A':
        # Check if form is bad (Poor or Fair)
        if metrics['mp_classification'] in ['Poor', 'Fair']:
            if not bad_form_active:
                bad_form_active = True
                bad_form_timer = 0.0
            bad_form_timer += delta_time
        else:
            # Good form - reset timer
            bad_form_active = False
            bad_form_timer = 0.0
    
    # ======================================================================
    
    # ==========================================================================
    # BAD FORM ALERT - Center Screen Warning
    # ==========================================================================
    if bad_form_active and bad_form_timer > 0:
        # Alert box parameters
        alert_w = 500
        alert_h = 120
        alert_x = (w - alert_w) // 2
        alert_y = 150
        
        # Pulsing effect based on timer (more urgent as time increases)
        pulse = int(abs(np.sin(bad_form_timer * 3) * 30))  # Pulsing between 0-30
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (alert_x, alert_y), (alert_x + alert_w, alert_y + alert_h), 
                     (0, 0, 200 + pulse), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border (thicker as time increases)
        border_thickness = min(5 + int(bad_form_timer), 10)
        cv2.rectangle(frame, (alert_x, alert_y), (alert_x + alert_w, alert_y + alert_h), 
                     (0, 0, 255), border_thickness)
        
        # Alert icon (⚠)
        cv2.putText(frame, "⚠", (alert_x + 20, alert_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
        
        # Main warning text
        cv2.putText(frame, "BAD FORM DETECTED!", 
                   (alert_x + 100, alert_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Timer display
        timer_text = f"Duration: {int(bad_form_timer)}s"
        cv2.putText(frame, timer_text, 
                   (alert_x + 100, alert_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Corrective message
        cv2.putText(frame, "Correct your posture now!", 
                   (alert_x + 100, alert_y + 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ==========================================================================
    # DISPLAY UI - HYBRID METRICS
    # ==========================================================================
    
    # Exercise info
    cv2.putText(frame, f"Exercise: {selected_exercise.replace('_', ' ').title()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}  |  Stage: {stage}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # MediaPipe Assessment (Left side)
    y_offset = 100
    cv2.putText(frame, "Analysis", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
    cv2.putText(frame, f"Quality: {metrics['mp_classification']}", 
                (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {metrics['mp_confidence']:.1%}", 
                (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Angle metrics
    y_offset = 190
    cv2.putText(frame, "Angles:", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if metrics['hip_angle'] > 0:
        cv2.putText(frame, f"Hip: {metrics['hip_angle']}°", 
                    (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if metrics['arm_angle'] > 0:
        cv2.putText(frame, f"Arm: {metrics['arm_angle']}°", 
                    (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if metrics['knee_angle'] > 0:
        cv2.putText(frame, f"Knee: {metrics['knee_angle']}°", 
                    (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if metrics['back_angle'] > 0:
        cv2.putText(frame, f"Back: {metrics['back_angle']}°", 
                    (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # YOLO Assessment (Right side) - if available
    if yolo_available:
        panel_x = w - 280
        y_offset = 100
        
        status_color = (0, 255, 0) if metrics['yolo_detected'] else (0, 0, 255)
        # cv2.putText(frame, "Assessment", 
        #             (panel_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        
        # cv2.putText(frame, f"Status: {'DETECTED' if metrics['yolo_detected'] else 'NO DETECTION'}", 
        #             (panel_x, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        if metrics['yolo_detected']:
            cv2.putText(frame, f"Detection: {metrics['yolo_detection_conf']:.1%}", 
                        (panel_x, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Keypoint Avg: {metrics['yolo_keypoint_conf']:.1%}", 
                        (panel_x, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Quality: {metrics['yolo_keypoint_quality']}", 
                        (panel_x, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Additional info
            cv2.putText(frame, "Your trained model's", 
                        (panel_x, y_offset + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(frame, "confidence in detection", 
                        (panel_x, y_offset + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow('Workout Buddy - Hybrid MediaPipe + YOLO', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 70}")
print("  WORKOUT SUMMARY")
print(f"{'=' * 70}")
print(f"Exercise: {selected_exercise.replace('_', ' ').title()}")
print(f"Total Reps: {rep_count}")
print(f"\nAssessment:")
print(f"  Form Quality: {metrics['mp_classification']}")
print(f"  Confidence: {metrics['mp_confidence']:.1%}")
if yolo_available:
    print(f"\nYOLO Assessment:")
    print(f"  Detection Status: {'Active' if metrics['yolo_detected'] else 'Not Detected'}")
    if metrics['yolo_detected']:
        print(f"  Detection Confidence: {metrics['yolo_detection_conf']:.1%}")
        print(f"  Keypoint Quality: {metrics['yolo_keypoint_quality']}")
print("\nGreat work! 💪")
