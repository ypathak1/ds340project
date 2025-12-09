

import numpy as np
import cv2
import os
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

print("=" * 80)
print("DS340 GESTURE CALCULATOR - MODEL EVALUATION")
print("=" * 80)
print()

print("This script evaluates:")
print("  - Custom CNN (personalized model)")
print("  - MediaPipe Face Mesh (landmark-based rules)")
print("  - Ensemble (majority voting)")
print()
print("Metrics calculated:")
print("  - Accuracy, Precision, Recall, F1-Score")
print("  - Ensemble voting patterns and agreement")
print("  - Inference Time")
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model imports
try:
    from keras.models import load_model
    if os.path.exists("custom_emotion_model.h5"):
        custom_model = load_model("custom_emotion_model.h5")
        custom_labels = np.load("custom_emotion_labels.npy", allow_pickle=True)
        print("[INFO] Custom model loaded successfully")
    else:
        custom_model = None
        custom_labels = None
        print("[WARNING] Custom model not found - run 3_train_emotions.py first")
except ImportError:
    custom_model = None
    custom_labels = None
    print("[WARNING] Keras not available")

# MediaPipe Face Mesh
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
print("[INFO] MediaPipe Face Mesh loaded")

EMOTIONS = ['happy', 'sad', 'neutral', 'angry']
EMOTION_COLORS = {
    'happy': '#FFD93D',
    'sad': '#6BCB77',
    'neutral': '#4D96FF',
    'angry': '#FF6B6B'
}

# Create output directory
os.makedirs('results', exist_ok=True)

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def mediapipe_emotion_from_landmarks(face_landmarks):
    """Rule-based emotion detection using facial geometry"""
    try:
        # Key landmarks for emotion detection
        left_mouth = face_landmarks[61]
        right_mouth = face_landmarks[291]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        left_brow = face_landmarks[70]
        right_brow = face_landmarks[300]
        nose = face_landmarks[168]

        # Calculate geometric features
        mouth_avg_y = (left_mouth.y + right_mouth.y) / 2
        lip_center_y = (upper_lip.y + lower_lip.y) / 2
        brow_avg_y = (left_brow.y + right_brow.y) / 2
        brow_distance = nose.y - brow_avg_y

        # Rule-based classification
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


def predict_custom(landmarks):
    """Predict emotion using custom CNN model"""
    if not custom_model or not landmarks:
        return None
    try:
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        features = np.array(features).reshape(1, -1)
        probs = custom_model.predict(features, verbose=0)[0]
        return custom_labels[np.argmax(probs)]
    except:
        return None


def predict_mediapipe(landmarks):
    """Predict emotion using MediaPipe geometric rules"""
    if not landmarks:
        return None
    try:
        return mediapipe_emotion_from_landmarks(landmarks)
    except:
        return None


def predict_ensemble(landmarks):
    """Predict emotion using majority voting (Custom + MediaPipe only)"""
    votes = []

    custom_pred = predict_custom(landmarks)
    if custom_pred:
        votes.append(custom_pred)

    mp_pred = predict_mediapipe(landmarks)
    if mp_pred:
        votes.append(mp_pred)

    if votes:
        return Counter(votes).most_common(1)[0][0]
    return 'neutral'


# =============================================================================
# LOAD TEST DATA
# =============================================================================

print("\n[INFO] Loading test data...")

test_data = {emotion: [] for emotion in EMOTIONS}

for emotion in EMOTIONS:
    file_path = f"emotion_{emotion}.npy"
    if os.path.exists(file_path):
        data = np.load(file_path)
        # Use 20% for testing
        test_size = max(1, int(len(data) * 0.2))
        test_data[emotion] = data[-test_size:]
        print(f"  {emotion}: {len(test_data[emotion])} test samples")

total_samples = sum(len(v) for v in test_data.values())
if total_samples == 0:
    print("\n[ERROR] No test data found!")
    print("Please run 1_collect_emotions.py to collect emotion data first.")
    exit(1)

print(f"\nTotal test samples: {total_samples}")

# =============================================================================
# RUN EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING EVALUATION")
print("=" * 80)

# Initialize result tracking
results = {
    'Custom': {'y_true': [], 'y_pred': [], 'times': []},
    'MediaPipe': {'y_true': [], 'y_pred': [], 'times': []},
    'Ensemble': {'y_true': [], 'y_pred': [], 'times': []}
}

# Track predictions for voting timeline visualization
voting_timeline = {
    'Custom': [],
    'MediaPipe': [],
    'Ensemble': [],
    'ground_truth': []
}

# Store sample landmarks for visualization
sample_landmarks_by_emotion = {emotion: None for emotion in EMOTIONS}

print("\nProcessing test samples...")
sample_count = 0

for emotion, landmarks_list in test_data.items():
    if len(landmarks_list) == 0:
        continue

    print(f"\nEvaluating {emotion} samples...")

    for idx, landmark_data in enumerate(landmarks_list):
        sample_count += 1

        # Convert to landmark objects
        class Landmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        landmarks = []
        for i in range(0, len(landmark_data), 3):
            landmarks.append(Landmark(landmark_data[i], landmark_data[i+1], landmark_data[i+2]))

        # Save first sample of each emotion for visualization
        if sample_landmarks_by_emotion[emotion] is None:
            sample_landmarks_by_emotion[emotion] = landmarks

        # Evaluate Custom Model
        custom_pred = None
        if custom_model:
            start = time.time()
            custom_pred = predict_custom(landmarks)
            elapsed = time.time() - start
            if custom_pred:
                results['Custom']['y_true'].append(emotion)
                results['Custom']['y_pred'].append(custom_pred)
                results['Custom']['times'].append(elapsed)

        # Evaluate MediaPipe
        start = time.time()
        mp_pred = predict_mediapipe(landmarks)
        elapsed = time.time() - start
        if mp_pred:
            results['MediaPipe']['y_true'].append(emotion)
            results['MediaPipe']['y_pred'].append(mp_pred)
            results['MediaPipe']['times'].append(elapsed)

        # Evaluate Ensemble
        start = time.time()
        ensemble_pred = predict_ensemble(landmarks)
        elapsed = time.time() - start
        results['Ensemble']['y_true'].append(emotion)
        results['Ensemble']['y_pred'].append(ensemble_pred)
        results['Ensemble']['times'].append(elapsed)

        # Track ALL samples for agreement (not just first 100)
        voting_timeline['Custom'].append(custom_pred if custom_pred else 'none')
        voting_timeline['MediaPipe'].append(mp_pred if mp_pred else 'none')
        voting_timeline['Ensemble'].append(ensemble_pred)
        voting_timeline['ground_truth'].append(emotion)

        if sample_count % 10 == 0:
            print(f"  Processed {sample_count} samples...")

print(f"\nEvaluation complete: {sample_count} samples processed")

# =============================================================================
# CALCULATE METRICS
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING METRICS")
print("=" * 80)

metrics_summary = {}

for model_name, data in results.items():
    if len(data['y_pred']) == 0:
        print(f"\n{model_name}: No predictions available")
        continue

    y_true = data['y_true']
    y_pred = data['y_pred']

    # Calculate overall metrics
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
        'avg_time': avg_time
    }

# Print summary
print("\nModel Performance Summary:")
print("-" * 80)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 80)
for model_name, metrics in metrics_summary.items():
    print(f"{model_name:<15} "
          f"{metrics['accuracy']*100:>10.2f}% "
          f"{metrics['precision']*100:>10.2f}% "
          f"{metrics['recall']*100:>10.2f}% "
          f"{metrics['f1']*100:>10.2f}%")
print("-" * 80)

# =============================================================================
# VISUALIZATION 1: VOTING TIMELINE (FIRST 100 SAMPLES ONLY)
# =============================================================================

print("\n[INFO] Generating visualizations...")
print("  [1/3] Creating Voting Timeline...")

plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(16, 8))

# Map emotions to y-positions
emotion_to_y = {emotion: i for i, emotion in enumerate(EMOTIONS)}
emotion_to_y['none'] = -1

# Use ONLY first 100 for timeline clarity
timeline_len = min(100, len(voting_timeline['ground_truth']))
x_vals = list(range(timeline_len))

# Plot each model's predictions (first 100)
models_to_plot = ['Custom', 'MediaPipe']
model_colors = ['#4ECDC4', '#45B7D1']
model_markers = ['s', '^']

for model_idx, model_name in enumerate(models_to_plot):
    y_vals = [emotion_to_y[voting_timeline[model_name][i]] for i in range(timeline_len)]
    ax.scatter(x_vals, y_vals,
              c=model_colors[model_idx],
              marker=model_markers[model_idx],
              s=30,
              alpha=0.6,
              label=model_name,
              zorder=3)

# Plot ensemble decision
ensemble_y = [emotion_to_y[voting_timeline['Ensemble'][i]] for i in range(timeline_len)]
ax.scatter(x_vals, ensemble_y,
          c='#96CEB4',
          marker='*',
          s=120,
          alpha=0.9,
          label='Ensemble Decision',
          edgecolors='black',
          linewidths=0.5,
          zorder=4)

# Background colors for ground truth
for i in range(timeline_len):
    true_emotion = voting_timeline['ground_truth'][i]
    rect = Rectangle((i-0.5, -1.5), 1, len(EMOTIONS)+1.5,
                     facecolor=EMOTION_COLORS[true_emotion],
                     alpha=0.1,
                     zorder=1)
    ax.add_patch(rect)

ax.set_yticks(list(range(len(EMOTIONS))))
ax.set_yticklabels(EMOTIONS)
ax.set_xlabel('Sample Number', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Emotion', fontsize=13, fontweight='bold')
ax.set_title('Ensemble Voting Timeline (First 100 Samples)\n(Background color indicates ground truth)',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='x')
ax.set_ylim(-0.5, len(EMOTIONS)-0.5)

plt.tight_layout()
plt.savefig('results/figure1_voting_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: results/figure1_voting_timeline.png")

# =============================================================================
# VISUALIZATION 2: MODEL AGREEMENT (ALL DATA)
# =============================================================================

print("  [2/3] Creating Model Agreement Analysis...")

# Calculate agreement using ALL samples (not just first 100)
total_samples = len(voting_timeline['ground_truth'])
agreement_counts = {'All Agree': 0, '2 Agree': 0, 'All Disagree': 0}
agreement_by_emotion = {emotion: {'All Agree': 0, '2 Agree': 0, 'All Disagree': 0, 'Total': 0}
                       for emotion in EMOTIONS}

for i in range(total_samples):
    true_emo = voting_timeline['ground_truth'][i]
    predictions = [
        voting_timeline['Custom'][i],
        voting_timeline['MediaPipe'][i]
    ]

    # Remove 'none' predictions
    valid_preds = [p for p in predictions if p != 'none']

    if len(valid_preds) >= 2:
        unique_preds = len(set(valid_preds))

        if unique_preds == 1:
            agreement_counts['All Agree'] += 1
            agreement_by_emotion[true_emo]['All Agree'] += 1
        else:
            agreement_counts['2 Agree'] += 1  # They disagree but both voted
            agreement_by_emotion[true_emo]['2 Agree'] += 1

        agreement_by_emotion[true_emo]['Total'] += 1

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Overall agreement pie chart
colors_pie = ['#96CEB4', '#FFD93D']
labels_pie = ['Both Agree', 'Disagree']
values_pie = [agreement_counts['All Agree'], agreement_counts['2 Agree']]
explode = (0.1, 0)

wedges, texts, autotexts = ax1.pie(
    values_pie,
    labels=labels_pie,
    autopct='%1.1f%%',
    colors=colors_pie,
    explode=explode,
    shadow=True,
    startangle=90
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
ax1.set_title('Overall Model Agreement\n(Custom + MediaPipe)', fontsize=14, fontweight='bold', pad=20)

# Right: Agreement breakdown by emotion (ALL emotions with data)
emotions_present = [e for e in EMOTIONS if agreement_by_emotion[e]['Total'] > 0]
x_pos = np.arange(len(emotions_present))
width = 0.35

agree_vals = []
disagree_vals = []

for emotion in emotions_present:
    total = agreement_by_emotion[emotion]['Total']
    if total > 0:
        agree_vals.append(agreement_by_emotion[emotion]['All Agree'] / total * 100)
        disagree_vals.append(agreement_by_emotion[emotion]['2 Agree'] / total * 100)

bars1 = ax2.bar(x_pos - width/2, agree_vals, width, label='Both Agree', color='#96CEB4')
bars2 = ax2.bar(x_pos + width/2, disagree_vals, width, label='Disagree', color='#FFD93D')

ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Ground Truth Emotion', fontsize=12, fontweight='bold')
ax2.set_title('Model Agreement by Emotion\n(All Test Data)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(emotions_present)
ax2.legend()
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figure2_model_agreement.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: results/figure2_model_agreement.png")

# =============================================================================
# VISUALIZATION 3: MODEL COMPARISON
# =============================================================================

print("  [3/3] Creating Model Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Overall accuracy comparison
models = []
accuracies = []
colors_bars = ['#4ECDC4', '#95E1D3', '#2C3E50']

for model_name in ['Custom', 'MediaPipe', 'Ensemble']:
    if model_name in metrics_summary:
        models.append(model_name)
        accuracies.append(metrics_summary[model_name]['accuracy'] * 100)

bars = ax1.bar(models, accuracies, color=colors_bars[:len(models)],
               edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Overall Model Accuracy', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=25, color='red', linestyle=':', alpha=0.5, linewidth=1, label='Random (25%)')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=11)

ax1.legend()

# Right: Per-class accuracy for Ensemble
if 'Ensemble' in metrics_summary:
    ensemble_data = results['Ensemble']
    y_true = np.array(ensemble_data['y_true'])
    y_pred = np.array(ensemble_data['y_pred'])

    per_class_acc = []
    for emotion in EMOTIONS:
        mask = y_true == emotion
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == emotion).sum() / mask.sum() * 100
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0)

    emotion_colors_list = ['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B6B']
    bars2 = ax2.bar(EMOTIONS, per_class_acc, color=emotion_colors_list,
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Emotion Class', fontsize=13, fontweight='bold')
    ax2.set_title('Ensemble Per-Class Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

plt.suptitle('Model Performance Comparison',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figure3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: results/figure3_model_comparison.png")

# =============================================================================
# GENERATE TEXT REPORT
# =============================================================================

print("\n[INFO] Generating evaluation report...")

with open('results/evaluation_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GESTURE CALCULATOR - EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("OVERALL RESULTS\n")
    f.write("-" * 80 + "\n\n")

    f.write(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-" * 80 + "\n")

    for model_name, metrics in metrics_summary.items():
        f.write(f"{model_name:<15} "
                f"{metrics['accuracy']*100:>10.2f}%  "
                f"{metrics['precision']*100:>10.2f}%  "
                f"{metrics['recall']*100:>10.2f}%  "
                f"{metrics['f1']*100:>10.2f}%\n")

    f.write("\n" + "=" * 80 + "\n\n")

    # Per-emotion breakdown for Ensemble
    if 'Ensemble' in metrics_summary:
        f.write("ENSEMBLE MODEL - PER-EMOTION BREAKDOWN\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")

        ensemble = metrics_summary['Ensemble']
        for i, emotion in enumerate(EMOTIONS):
            f.write(f"{emotion.capitalize():<12} "
                    f"{ensemble['precision_per_class'][i]*100:>10.2f}%  "
                    f"{ensemble['recall_per_class'][i]*100:>10.2f}%  "
                    f"{ensemble['f1_per_class'][i]*100:>10.2f}%\n")

    f.write("\n" + "=" * 80 + "\n\n")

    # Model agreement statistics
    f.write("MODEL AGREEMENT STATISTICS\n")
    f.write("-" * 80 + "\n\n")

    total_agree = sum(agreement_counts.values())
    for category, count in agreement_counts.items():
        percentage = (count / total_agree * 100) if total_agree > 0 else 0
        f.write(f"{category:<20} {count:>5} samples ({percentage:>5.1f}%)\n")

    f.write("\n" + "=" * 80 + "\n\n")

    # Performance metrics
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 80 + "\n\n")

    f.write(f"{'Model':<15} {'Avg Inference Time':<20}\n")
    f.write("-" * 80 + "\n")

    for model_name, metrics in metrics_summary.items():
        f.write(f"{model_name:<15} {metrics['avg_time']*1000:>18.2f} ms\n")

    f.write("\n" + "=" * 80 + "\n\n")

    # Key findings
    f.write("KEY FINDINGS\n")
    f.write("-" * 80 + "\n\n")

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
            f.write(f"3. Ensemble improvement: {improvement:+.2f}%\n")

            # Agreement statistics
            all_agree_pct = (agreement_counts['All Agree'] / total_agree * 100) if total_agree > 0 else 0
            f.write(f"4. Models agree {all_agree_pct:.1f}% of the time\n")

            if improvement > 0:
                f.write("5. Ensemble approach shows improvement\n")
            else:
                f.write("5. Ensemble performance similar to best single model\n")

            # Inference time
            total_time = metrics_summary['Ensemble']['avg_time']
            f.write(f"6. Ensemble inference time: {total_time*1000:.2f}ms (real-time capable)\n")

    f.write("\n" + "=" * 80 + "\n")

print("    Saved: results/evaluation_report.txt")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print("\nGenerated Visualizations:")
print("-" * 80)
print("  Figure 1: results/figure1_voting_timeline.png")
print("    - Ensemble voting behavior (first 100 samples)")
print("  Figure 2: results/figure2_model_agreement.png")
print("    - Model agreement analysis (all test data)")
print("  Figure 3: results/figure3_model_comparison.png")
print("    - Overall and per-class accuracy comparison")
print("  Report: results/evaluation_report.txt")
print("    - Complete metrics and findings")
print("\nNote: FER was not evaluated as it requires image data, not landmarks.")
print("=" * 80 + "\n")

# Cleanup
face_mesh.close()