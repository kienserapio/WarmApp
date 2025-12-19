import cv2
import mediapipe as mp
import numpy as np

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
    
    This simulates model predictions using biomechanically sound angle ranges.
    """
    total_score = 0
    count = 0
    
    for angle, (min_val, max_val) in zip(angles, ideal_ranges):
        if min_val <= angle <= max_val:
            # Perfect form
            score = 1.0
        else:
            # Calculate how far off from ideal range
            if angle < min_val:
                deviation = min_val - angle
            else:
                deviation = angle - max_val
            
            # Penalize based on deviation (max penalty at 30 degrees off)
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

print("=== Workout Buddy - AI-Powered Exercise Tracker ===")
print("Select exercise to track:")
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
print("✓ AI Form Scoring Active")
print("Press ESC to exit\n")

# Define keypoint triplets and ideal ranges for each exercise
# MediaPipe Pose Landmarks: https://google.github.io/mediapipe/solutions/pose.html

# Push-up keypoints and ideal ranges
pushup_hips = [(11, 23, 25), (12, 24, 26)]  # shoulder-hip-knee
pushup_arms = [(12, 14, 16), (11, 13, 15)]  # shoulder-elbow-wrist
pushup_ideal_hip = (160, 200)  # Straight body line
pushup_ideal_arm_down = (80, 110)  # 90-degree elbow bend
pushup_ideal_arm_up = (160, 180)  # Arms extended

# Glute Bridge keypoints
glute_bridge_hips = [(11, 23, 25), (12, 24, 26)]
glute_bridge_knees = [(23, 25, 27), (24, 26, 28)]
glute_bridge_ideal_hip = (160, 200)  # Hip extension
glute_bridge_ideal_knee = (80, 110)  # 90-degree knees

# Good Morning keypoints
good_morning_back = [(11, 23, 25), (12, 24, 26)]
good_morning_knees = [(23, 25, 27), (24, 26, 28)]
good_morning_ideal_back_down = (45, 90)  # Hip hinge
good_morning_ideal_back_up = (150, 180)  # Standing
good_morning_ideal_knee = (160, 200)  # Straight legs

# Plank keypoints
plank_back = [(11, 23, 25), (12, 24, 26)]
plank_shoulders = [(13, 11, 23), (14, 12, 24)]
plank_ideal_back = (160, 200)  # Straight line
plank_ideal_shoulder = (80, 110)  # Proper shoulder position

# Pull-up keypoints
pullup_arms = [(12, 14, 16), (11, 13, 15)]
pullup_shoulders = [(14, 12, 24), (13, 11, 23)]
pullup_ideal_arm_up = (60, 90)  # Flexed at top
pullup_ideal_arm_down = (160, 180)  # Extended at bottom

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Counters and metrics
rep_count = 0
stage = "up" if selected_exercise != 'plank' else "inactive"
metrics = {
    'ai_classification': 'N/A',
    'ai_confidence': 0.0,
    'form_quality': 'N/A',
    'hip_angle': 0,
    'arm_angle': 0,
    'knee_angle': 0,
    'back_angle': 0
}

# Frame counter for smoother updates
frame_counter = 0
update_interval = 5  # Update AI prediction every 5 frames

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        # Draw MediaPipe skeleton
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
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
            
            # AI Form Scoring
            if frame_counter % update_interval == 0:
                avg_arm_angle = np.mean(arm_angles)
                if avg_arm_angle < 130:  # Down position
                    ideal_ranges = [pushup_ideal_hip, pushup_ideal_arm_down]
                    angles = [metrics['hip_angle'], metrics['arm_angle']]
                else:  # Up position
                    ideal_ranges = [pushup_ideal_hip, pushup_ideal_arm_up]
                    angles = [metrics['hip_angle'], metrics['arm_angle']]
                
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            # Count reps
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
            
            # AI Form Scoring
            if frame_counter % update_interval == 0:
                ideal_ranges = [glute_bridge_ideal_hip, glute_bridge_ideal_knee]
                angles = [metrics['hip_angle'], metrics['knee_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            # Count reps
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
            
            # AI Form Scoring
            if frame_counter % update_interval == 0:
                avg_back_angle = np.mean(back_angles)
                if avg_back_angle < 120:  # Hinged position
                    ideal_ranges = [good_morning_ideal_back_down, good_morning_ideal_knee]
                else:  # Standing
                    ideal_ranges = [good_morning_ideal_back_up, good_morning_ideal_knee]
                angles = [metrics['back_angle'], metrics['knee_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            # Count reps
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
            
            # AI Form Scoring
            if frame_counter % update_interval == 0:
                ideal_ranges = [plank_ideal_back, plank_ideal_shoulder]
                angles = [metrics['back_angle'], metrics['arm_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            # Track hold time
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
            
            # AI Form Scoring
            if frame_counter % update_interval == 0:
                avg_arm_angle = np.mean(arm_angles)
                if avg_arm_angle < 120:  # Up position
                    ideal_ranges = [pullup_ideal_arm_up]
                else:  # Down position
                    ideal_ranges = [pullup_ideal_arm_down]
                angles = [metrics['arm_angle']]
                classification, confidence = calculate_form_confidence(angles, ideal_ranges)
                metrics['ai_classification'] = classification
                metrics['ai_confidence'] = confidence
            
            # Count reps
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle < 90 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_arm_angle > 160:
                stage = "down"
        
        frame_counter += 1
        
        # ==================== DISPLAY METRICS ====================
        cv2.putText(frame, f"Exercise: {selected_exercise.replace('_', ' ').title()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Reps: {rep_count}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Stage: {stage}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # AI predictions
        y_offset = 120
        cv2.putText(frame, "=== AI Form Analysis ===", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Quality: {metrics['ai_classification']}", 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {metrics['ai_confidence']:.1%}", 
                    (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Angle metrics
        y_offset = 210
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

    cv2.imshow('Workout Buddy - AI Powered', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n=== Workout Summary ===")
print(f"Exercise: {selected_exercise.replace('_', ' ').title()}")
print(f"Total Reps: {rep_count}")
print(f"Final AI Classification: {metrics['ai_classification']}")
print(f"Final Confidence: {metrics['ai_confidence']:.1%}")
print("Keep up the great work!")
