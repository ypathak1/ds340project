"""
STEP 6: COMPREHENSIVE MODEL EVALUATION
Evaluates all models (FER, Custom, MediaPipe, Ensemble) with proper metrics
Generates confusion matrices, accuracy reports, and performance comparisons

"""

import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model
from fer import FER
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time

print("=" * 70)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)
print("\nThis script evaluates:")
print("  1. FER Pretrained Model")
print("  2. Custom Trained Model")
print("  3. MediaPipe Landmarks Model")
print("  4. Ensemble (All 3 Combined)")
print("\nMetrics calculated:")
print("  - Accuracy")
print("  - Precision, Recall, F1-Score per emotion")
print("  - Confusion Matrices")
print("  - Inference Time")
print()

# Create results directory
os.makedirs("results", exist_ok=True)

# SETUP MODELS
print("[INFO] Loading models...")

# FER
fer_detector = None
try:
    fer_detector = FER(mtcnn=False)
    print("  ✓ FER loaded")
except:
    print("  ✗ FER not available")

# Custom Model
custom_model = None
custom_labels = None
if os.path.exists("custom_emotion_model.h5"):
    try:
        custom_model = load_model("custom_emotion_model.h5")
        custom_labels = np.load("custom_emotion_labels.npy")
        print("  ✓ Custom model loaded")
    except:
        print("  ✗ Custom model failed to load")
else:
    print("  ✗ Custom model not found (run 3_train_emotions.py first)")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("  ✓ MediaPipe loaded")

# Mappings
FER_MAPPING = {
    'happy': 'happy', 'sad': 'sad', 'neutral': 'neutral', 'angry': 'angry',
    'surprise': 'happy', 'fear': 'sad', 'disgust': 'angry'
}

EMOTIONS = ['happy', 'sad', 'neutral', 'angry']

# MEDIAPIPE EMOTION DETECTION
def mediapipe_emotion_from_landmarks(face_landmarks):
    """Detect emotion using MediaPipe face landmarks"""
    try:
        left_mouth = face_landmarks[61]
        right_mouth = face_landmarks[291]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        left_brow = face_landmarks[70]
        right_brow = face_landmarks[300]
        nose = face_landmarks[168]
        
        mouth_avg_y = (left_mouth.y + right_mouth.y) / 2
        lip_center_y = (upper_lip.y + lower_lip.y) / 2
        brow_avg_y = (left_brow.y + right_brow.y) / 2
        brow_distance = nose.y - brow_avg_y
        
        if mouth_avg_y < lip_center_y - 0.015:
            return 'happy'
        elif mouth_avg_y > lip_center_y + 0.01:
            return 'sad'
        elif brow_distance < 0.06:
            return 'angry'
        else:
            return 'neutral'
    except:
        return 'neutral'

# LOAD TEST DATA
print("\n[INFO] Loading test data...")

test_data = {'happy': [], 'sad': [], 'neutral': [], 'angry': []}
test_images = {'happy': [], 'sad': [], 'neutral': [], 'angry': []}

# Load .npy files
for emotion in EMOTIONS:
    file_path = f"emotion_{emotion}.npy"
    if os.path.exists(file_path):
        data = np.load(file_path)
        # Use last 20% as test set
        test_size = int(len(data) * 0.2)
        if test_size > 0:
            test_data[emotion] = data[-test_size:]
            print(f"  ✓ {emotion}: {len(test_data[emotion])} test samples")
        else:
            print(f"  ⚠ {emotion}: Not enough data")
    else:
        print(f"  ✗ {emotion}: No data file found")

# Check if we have enough test data
total_test = sum(len(v) for v in test_data.values())
if total_test < 20:
    print("\n[ERROR] Not enough test data!")
    print("Please run 1_collect_emotions.py to collect at least 50 samples per emotion")
    exit(1)

print(f"\n[INFO] Total test samples: {total_test}")

# EVALUATION FUNCTIONS

def predict_fer(frame):
    """Predict emotion using FER"""
    if not fer_detector:
        return None
    try:
        result = fer_detector.detect_emotions(frame)
        if result:
            emo_dict = result[0]['emotions']
            fer_emo = max(emo_dict, key=emo_dict.get)
            return FER_MAPPING.get(fer_emo, 'neutral')
    except:
        pass
    return None

def predict_custom(landmarks):
    """Predict emotion using custom model"""
    if not custom_model:
        return None
    try:
        feats = []
        for lm in landmarks:
            feats.extend([lm.x, lm.y, lm.z])
        feats = np.array(feats).reshape(1, -1)
        probs = custom_model.predict(feats, verbose=0)[0]
        return custom_labels[np.argmax(probs)]
    except:
        return None

def predict_mediapipe(landmarks):
    """Predict emotion using MediaPipe landmarks"""
    try:
        return mediapipe_emotion_from_landmarks(landmarks)
    except:
        return None

def predict_ensemble(frame, landmarks):
    """Predict emotion using ensemble voting"""
    votes = []
    
    # FER vote
    fer_pred = predict_fer(frame)
    if fer_pred:
        votes.append(fer_pred)
    
    # Custom vote
    custom_pred = predict_custom(landmarks)
    if custom_pred:
        votes.append(custom_pred)
    
    # MediaPipe vote
    mp_pred = predict_mediapipe(landmarks)
    if mp_pred:
        votes.append(mp_pred)
    
    if votes:
        return Counter(votes).most_common(1)[0][0]
    return 'neutral'

# GENERATE TEST IMAGES FROM LANDMARKS
print("\n[INFO] Generating test images from landmark data...")
print("(This simulates what the webcam would capture)")

def landmarks_to_image(landmark_data):
    """Convert landmark data back to a simple visualization for FER"""
    # Create blank image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Draw face mesh points (simplified)
    for i in range(0, len(landmark_data), 3):
        x = int(landmark_data[i] * 224)
        y = int(landmark_data[i+1] * 224)
        if 0 <= x < 224 and 0 <= y < 224:
            cv2.circle(img, (x, y), 1, (0, 0, 0), -1)
    
    return img

# RUN EVALUATION
print("\n" + "=" * 70)
print("RUNNING EVALUATION")
print("=" * 70)

results = {
    'FER': {'y_true': [], 'y_pred': [], 'times': []},
    'Custom': {'y_true': [], 'y_pred': [], 'times': []},
    'MediaPipe': {'y_true': [], 'y_pred': [], 'times': []},
    'Ensemble': {'y_true': [], 'y_pred': [], 'times': []}
}

print("\nTesting each sample...")
sample_count = 0

for emotion, landmarks_list in test_data.items():
    if len(landmarks_list) == 0:
        continue
    
    print(f"\nEvaluating {emotion} samples...")
    
    for landmark_data in landmarks_list:
        sample_count += 1
        
        # Convert to image for FER
        img = landmarks_to_image(landmark_data)
        
        # Convert to landmark format for other models
        landmarks = []
        for i in range(0, len(landmark_data), 3):
            class Landmark:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            landmarks.append(Landmark(landmark_data[i], landmark_data[i+1], landmark_data[i+2]))
        
        # Test FER
        if fer_detector:
            start = time.time()
            pred = predict_fer(img)
            elapsed = time.time() - start
            if pred:
                results['FER']['y_true'].append(emotion)
                results['FER']['y_pred'].append(pred)
                results['FER']['times'].append(elapsed)
        
        # Test Custom
        if custom_model:
            start = time.time()
            pred = predict_custom(landmarks)
            elapsed = time.time() - start
            if pred:
                results['Custom']['y_true'].append(emotion)
                results['Custom']['y_pred'].append(pred)
                results['Custom']['times'].append(elapsed)
        
        # Test MediaPipe
        start = time.time()
        pred = predict_mediapipe(landmarks)
        elapsed = time.time() - start
        if pred:
            results['MediaPipe']['y_true'].append(emotion)
            results['MediaPipe']['y_pred'].append(pred)
            results['MediaPipe']['times'].append(elapsed)
        
        # Test Ensemble
        start = time.time()
        pred = predict_ensemble(img, landmarks)
        elapsed = time.time() - start
        results['Ensemble']['y_true'].append(emotion)
        results['Ensemble']['y_pred'].append(pred)
        results['Ensemble']['times'].append(elapsed)
        
        if sample_count % 10 == 0:
            print(f"  Processed {sample_count} samples...")

print(f"\n✓ Evaluation complete: {sample_count} samples tested")

# CALCULATE METRICS
print("\n" + "=" * 70)
print("CALCULATING METRICS")
print("=" * 70)

metrics_summary = {}

for model_name, data in results.items():
    if len(data['y_pred']) == 0:
        print(f"\n{model_name}: No predictions available")
        continue
    
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=EMOTIONS, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=EMOTIONS, average=None, zero_division=0
    )
    
    # Inference time
    avg_time = np.mean(data['times']) if data['times'] else 0
    
    metrics_summary[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'avg_time': avg_time,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    }

# GENERATE VISUALIZATIONS
print("\n[INFO] Generating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Confusion Matrices
for model_name, metrics in metrics_summary.items():
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {metrics["accuracy"]*100:.1f}%',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.tight_layout()
    
    filename = f'results/confusion_matrix_{model_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filename}")

# 2. Model Comparison - Accuracy
plt.figure(figsize=(10, 6))
models = list(metrics_summary.keys())
accuracies = [metrics_summary[m]['accuracy'] * 100 for m in models]

bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 100])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/model_comparison_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved results/model_comparison_accuracy.png")

# 3. Per-Emotion Performance (Ensemble)
if 'Ensemble' in metrics_summary:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(EMOTIONS))
    width = 0.25
    
    ensemble_metrics = metrics_summary['Ensemble']
    
    bars1 = ax.bar(x - width, ensemble_metrics['precision_per_class'] * 100, width, 
                   label='Precision', color='#FF6B6B')
    bars2 = ax.bar(x, ensemble_metrics['recall_per_class'] * 100, width,
                   label='Recall', color='#4ECDC4')
    bars3 = ax.bar(x + width, ensemble_metrics['f1_per_class'] * 100, width,
                   label='F1-Score', color='#45B7D1')
    
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ensemble Model - Per-Emotion Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS)
    ax.legend()
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/per_emotion_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved results/per_emotion_performance.png")

# 4. Inference Time Comparison
plt.figure(figsize=(10, 6))
models = list(metrics_summary.keys())
times = [metrics_summary[m]['avg_time'] * 1000 for m in models]  # Convert to ms

bars = plt.bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.ylabel('Inference Time (ms)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.title('Average Inference Time Comparison', fontsize=14, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}ms', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('results/inference_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved results/inference_time_comparison.png")

# GENERATE TEXT REPORT
print("\n[INFO] Generating evaluation report...")

with open('results/evaluation_report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("AI MATH CALCULATOR - EVALUATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("OVERALL RESULTS\n")
    f.write("-" * 70 + "\n\n")
    
    # Summary table
    f.write(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-" * 70 + "\n")
    
    for model_name, metrics in metrics_summary.items():
        f.write(f"{model_name:<15} "
                f"{metrics['accuracy']*100:>10.2f}%  "
                f"{metrics['precision']*100:>10.2f}%  "
                f"{metrics['recall']*100:>10.2f}%  "
                f"{metrics['f1']*100:>10.2f}%\n")
    
    f.write("\n" + "=" * 70 + "\n\n")
    
    # Per-emotion breakdown for Ensemble
    if 'Ensemble' in metrics_summary:
        f.write("ENSEMBLE MODEL - PER-EMOTION BREAKDOWN\n")
        f.write("-" * 70 + "\n\n")
        
        f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 70 + "\n")
        
        ensemble = metrics_summary['Ensemble']
        for i, emotion in enumerate(EMOTIONS):
            f.write(f"{emotion.capitalize():<12} "
                    f"{ensemble['precision_per_class'][i]*100:>10.2f}%  "
                    f"{ensemble['recall_per_class'][i]*100:>10.2f}%  "
                    f"{ensemble['f1_per_class'][i]*100:>10.2f}%\n")
    
    f.write("\n" + "=" * 70 + "\n\n")
    
    # Performance metrics
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 70 + "\n\n")
    
    f.write(f"{'Model':<15} {'Avg Inference Time':<20}\n")
    f.write("-" * 70 + "\n")
    
    for model_name, metrics in metrics_summary.items():
        f.write(f"{model_name:<15} {metrics['avg_time']*1000:>18.2f} ms\n")
    
    f.write("\n" + "=" * 70 + "\n\n")
    
    # Key findings
    f.write("KEY FINDINGS\n")
    f.write("-" * 70 + "\n\n")
    
    if 'Ensemble' in metrics_summary:
        ensemble_acc = metrics_summary['Ensemble']['accuracy'] * 100
        
        # Find best single model
        single_models = {k: v for k, v in metrics_summary.items() if k != 'Ensemble'}
        if single_models:
            best_single = max(single_models.items(), key=lambda x: x[1]['accuracy'])
            best_single_acc = best_single[1]['accuracy'] * 100
            improvement = ensemble_acc - best_single_acc
            
            f.write(f"1. Ensemble achieves {ensemble_acc:.2f}% accuracy\n")
            f.write(f"2. Best single model: {best_single[0]} at {best_single_acc:.2f}%\n")
            f.write(f"3. Ensemble improvement: +{improvement:.2f}%\n")
            
            if improvement > 0:
                f.write(f"4. ✓ Ensemble approach is EFFECTIVE\n")
            
            # Total inference time
            total_time = sum(m['avg_time'] for m in metrics_summary.values() if m != metrics_summary['Ensemble'])
            f.write(f"5. Combined inference time: {total_time*1000:.2f}ms (still real-time!)\n")
    
    f.write("\n" + "=" * 70 + "\n")

print("  ✓ Saved results/evaluation_report.txt")

# PRINT SUMMARY TO TERMINAL
print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)

print("\nRESULTS SUMMARY:")
print("-" * 70)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)

for model_name, metrics in metrics_summary.items():
    print(f"{model_name:<15} "
          f"{metrics['accuracy']*100:>10.2f}%  "
          f"{metrics['precision']*100:>10.2f}%  "
          f"{metrics['recall']*100:>10.2f}%  "
          f"{metrics['f1']*100:>10.2f}%")

print("\n" + "=" * 70)
print("GENERATED FILES:")
print("-" * 70)
print("  ✓ results/confusion_matrix_*.png (4 files)")
print("  ✓ results/model_comparison_accuracy.png")
print("  ✓ results/per_emotion_performance.png")
print("  ✓ results/inference_time_comparison.png")
print("  ✓ results/evaluation_report.txt")
print("\nYou can now:")
print("  1. View PNG files in Preview/Photos")
print("  2. Read detailed report in evaluation_report.txt")
print("  3. Include these in your presentation/paper")
print("=" * 70 + "\n")

face_mesh.close()
