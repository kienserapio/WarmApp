import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import yaml
import os

# Load models function
def load_tensorflowjs_model(model_path):
    """Load TensorFlow.js model and metadata"""
    try:
        print(f"Loading model from: {model_path}")
        
        # Load the TensorFlow.js model
        model = tfjs.converters.load_keras_model(f"{model_path}/model.json")
        print("✓ Model loaded successfully")
        
        # Load metadata to get information about the model
        metadata_path = f"{model_path}/metadata.yaml"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            print(f"✓ Metadata loaded: {metadata.get('description', 'N/A')}")
        else:
            metadata = {}
            print("⚠ No metadata.yaml found")
        
        return model, metadata
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_frame_for_model(frame, target_size=(640, 640)):
    """Preprocess frame for YOLO pose model"""
    # Resize to model input size (YOLO uses 640x640)
    img = cv2.resize(frame, target_size)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def get_pose_classification(predictions, threshold=0.5):
    """Extract classification and confidence from model predictions"""
    # YOLO models output multiple values, we need to interpret them
    # For pose models, this typically includes confidence scores
    if predictions is None or len(predictions) == 0:
        return "Unknown", 0.0
    
    # Get the prediction with highest confidence
    # Adjust based on your model's actual output format
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    
    # For classification models, get argmax and confidence
    if len(predictions.shape) == 2 and predictions.shape[1] > 1:
        confidence = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))
        
        # Map to form quality labels (adjust based on your training)
        class_labels = ["Good Form", "Poor Form", "Neutral"]
        if class_idx < len(class_labels):
            return class_labels[class_idx], confidence
        return f"Class_{class_idx}", confidence
    
    # Single output (regression or binary)
    confidence = float(predictions[0][0]) if predictions.shape[1] == 1 else float(np.mean(predictions))
    if confidence > threshold:
        return "Good Form", confidence
    else:
        return "Poor Form", 1.0 - confidence
    
    return "Analyzing", confidence

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

print("=== Workout Buddy - Exercise Tracker ===")
print("Select exercise to track:")
print("1. Push-ups")
print("2. Glute Bridge")
print("3. Good Morning")
print("4. Plank")
print("5. Pull-ups")
exercise_choice = input("Enter choice (1-5): ")

exercise_map = {
    '1': ('pushup', '/Users/kien/Files/workout/models/WarmApp_PushUp_Model'),
    '2': ('glute_bridge', '/Users/kien/Files/workout/models/WarmApp_GluteBridge_Model'),
    '3': ('good_morning', '/Users/kien/Files/workout/models/WarmApp_GoodMorning_Model'),
    '4': ('plank', '/Users/kien/Files/workout/models/WarmApp_Plank_Medium_Model'),
    '5': ('pullup', '/Users/kien/Files/workout/models/WarmApp_PullUp_Model')
}

selected_exercise, model_path = exercise_map.get(exercise_choice, ('pushup', '/Users/kien/Files/workout/models/WarmApp_PushUp_Model'))

print(f"\nTracking: {selected_exercise.replace('_', ' ').title()}")
print("Loading AI model...")

# Load the exercise-specific model
model, metadata = load_tensorflowjs_model(model_path)

if model is not None:
    print("✓ AI model ready for inference\n")
else:
    print("⚠ Running without AI model (angle-based detection only)\n")

print("Press ESC to exit\n")

# Define keypoint triplets for each exercise
# MediaPipe Pose Landmarks: https://google.github.io/mediapipe/solutions/pose.html
# 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow
# 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee
# 27: left_ankle, 28: right_ankle

# Push-up keypoints
pushup_hips = [(11, 23, 25), (12, 24, 26)]  # shoulder-hip-knee (body alignment)
pushup_arms = [(12, 14, 16), (11, 13, 15)]  # shoulder-elbow-wrist (arm angle)

# Glute Bridge keypoints
glute_bridge_hips = [(11, 23, 25), (12, 24, 26)]  # shoulder-hip-knee (hip extension)
glute_bridge_knees = [(23, 25, 27), (24, 26, 28)]  # hip-knee-ankle (knee angle)

# Good Morning keypoints
good_morning_back = [(11, 23, 25), (12, 24, 26)]  # shoulder-hip-knee (hip hinge)
good_morning_knees = [(23, 25, 27), (24, 26, 28)]  # hip-knee-ankle (knee stability)

# Plank keypoints
plank_back = [(11, 23, 25), (12, 24, 26)]  # shoulder-hip-knee (back alignment)
plank_shoulders = [(13, 11, 23), (14, 12, 24)]  # elbow-shoulder-hip (shoulder position)

# Pull-up keypoints
pullup_arms = [(12, 14, 16), (11, 13, 15)]  # shoulder-elbow-wrist (arm flexion)
pullup_shoulders = [(14, 12, 24), (13, 11, 23)]  # elbow-shoulder-hip (shoulder engagement)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Counters and metrics
rep_count = 0
stage = "up" if selected_exercise != 'plank' else "inactive"
metrics = {
    'model_confidence': 0.0,
    'predicted_class': 'N/A',
    'form_quality': 'N/A',
    'hip_angle': 0,
    'arm_angle': 0,
    'knee_angle': 0,
    'back_angle': 0
}

# Frame counter for model prediction (don't predict every frame)
frame_counter = 0
prediction_interval = 10  # Predict every 10 frames

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
        
        # ==================== MODEL PREDICTION ====================
        # Predict every N frames to reduce computational load
        if frame_counter % prediction_interval == 0 and model is not None:
            try:
                # Preprocess frame for model input
                preprocessed = preprocess_frame_for_model(frame)
                
                # Run model inference
                predictions = model.predict(preprocessed, verbose=0)
                
                # Extract classification and confidence
                pred_class, confidence = get_pose_classification(predictions)
                metrics['predicted_class'] = pred_class
                metrics['model_confidence'] = confidence
                
            except Exception as e:
                print(f"Prediction error: {e}")
                metrics['predicted_class'] = 'Error'
                metrics['model_confidence'] = 0.0
        elif model is None:
            metrics['predicted_class'] = 'No Model'
            metrics['model_confidence'] = 0.0
        
        frame_counter += 1
        
        # ==================== PUSH-UP DETECTION ====================
        if selected_exercise == 'pushup':
            hip_angles = []
            arm_angles = []
            
            for hips in pushup_hips:
                points = [landmarks[i] for i in hips]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                hip_angles.append(angle)
                
                color = (0, 255, 0) if 160 <= angle <= 200 else (0, 0, 255)
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
                
                color = (0, 255, 0) if 80 <= angle <= 110 else (0, 255, 255)
                cv2.line(frame, coords[0], coords[1], color, 2)
                cv2.line(frame, coords[1], coords[2], color, 2)
                for coord in coords:
                    cv2.circle(frame, coord, 4, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            metrics['hip_angle'] = int(np.mean(hip_angles))
            metrics['arm_angle'] = int(np.mean(arm_angles))
            
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle > 160 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_arm_angle < 100:
                stage = "down"
            
            angle_form = "Good" if (160 <= metrics['hip_angle'] <= 200 and 80 <= metrics['arm_angle'] <= 110) else "Poor"
            metrics['form_quality'] = angle_form
        
        # ==================== GLUTE BRIDGE DETECTION ====================
        elif selected_exercise == 'glute_bridge':
            hip_angles = []
            knee_angles = []
            
            for hips in glute_bridge_hips:
                points = [landmarks[i] for i in hips]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                hip_angles.append(angle)
                
                color = (0, 255, 0) if 160 <= angle <= 200 else (0, 0, 255)
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
                
                color = (0, 255, 0) if 80 <= angle <= 110 else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['hip_angle'] = int(np.mean(hip_angles))
            metrics['knee_angle'] = int(np.mean(knee_angles))
            
            avg_hip_angle = np.mean(hip_angles)
            if avg_hip_angle > 160 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_hip_angle < 120:
                stage = "down"
            
            metrics['form_quality'] = "Good" if (160 <= metrics['hip_angle'] <= 200 and 80 <= metrics['knee_angle'] <= 110) else "Poor"
        
        # ==================== GOOD MORNING DETECTION ====================
        elif selected_exercise == 'good_morning':
            back_angles = []
            knee_angles = []
            
            for back in good_morning_back:
                points = [landmarks[i] for i in back]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                back_angles.append(angle)
                
                color = (0, 255, 0) if 45 <= angle <= 90 else (0, 0, 255)
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
                
                color = (0, 255, 0) if 160 <= angle <= 200 else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['back_angle'] = int(np.mean(back_angles))
            metrics['knee_angle'] = int(np.mean(knee_angles))
            
            avg_back_angle = np.mean(back_angles)
            if avg_back_angle > 150 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_back_angle < 90:
                stage = "down"
            
            metrics['form_quality'] = "Good" if (45 <= metrics['back_angle'] <= 90 and 160 <= metrics['knee_angle'] <= 200) else "Poor"
        
        # ==================== PLANK DETECTION ====================
        elif selected_exercise == 'plank':
            back_angles = []
            shoulder_angles = []
            
            for back in plank_back:
                points = [landmarks[i] for i in back]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                back_angles.append(angle)
                
                color = (0, 255, 0) if 160 <= angle <= 200 else (0, 0, 255)
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
                
                color = (0, 255, 0) if 80 <= angle <= 110 else (255, 0, 0)
                cv2.line(frame, coords[0], coords[1], color, 3)
                cv2.line(frame, coords[1], coords[2], color, 3)
                for coord in coords:
                    cv2.circle(frame, coord, 5, color, -1)
                cv2.putText(frame, f"{int(angle)}°", coords[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            metrics['back_angle'] = int(np.mean(back_angles))
            metrics['arm_angle'] = int(np.mean(shoulder_angles))
            
            avg_back_angle = np.mean(back_angles)
            if 160 <= avg_back_angle <= 200:
                if stage == "inactive":
                    stage = "holding"
                    rep_count += 1
            else:
                stage = "inactive"
            
            metrics['form_quality'] = "Good" if (160 <= metrics['back_angle'] <= 200) else "Poor"
        
        # ==================== PULL-UP DETECTION ====================
        elif selected_exercise == 'pullup':
            arm_angles = []
            shoulder_angles = []
            
            for arms in pullup_arms:
                points = [landmarks[i] for i in arms]
                coords = [(int(p.x * w), int(p.y * h)) for p in points]
                angle = calculate_angle(coords[0], coords[1], coords[2])
                arm_angles.append(angle)
                
                color = (0, 255, 0) if angle < 90 else (0, 255, 255)
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
            
            avg_arm_angle = np.mean(arm_angles)
            if avg_arm_angle < 90 and stage == "down":
                stage = "up"
                rep_count += 1
            elif avg_arm_angle > 160:
                stage = "down"
            
            metrics['form_quality'] = "Good" if metrics['arm_angle'] < 90 else "Poor"
        
        # ==================== DISPLAY METRICS ====================
        cv2.putText(frame, f"Exercise: {selected_exercise.replace('_', ' ').title()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Reps: {rep_count}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Stage: {stage}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Model predictions
        y_offset = 120
        cv2.putText(frame, "=== AI Model ===", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Prediction: {metrics['predicted_class']}", 
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {metrics['model_confidence']:.2%}", 
                    (10, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Angle metrics
        y_offset = 200
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
        
        form_color = (0, 255, 0) if metrics['form_quality'] == "Good" else (0, 0, 255)
        cv2.putText(frame, f"Form: {metrics['form_quality']}", 
                    (10, y_offset + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)

    cv2.imshow('Workout Buddy - AI Powered', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n=== Workout Summary ===")
print(f"Exercise: {selected_exercise.replace('_', ' ').title()}")
print(f"Total Reps: {rep_count}")
print(f"Final Form Quality: {metrics['form_quality']}")
