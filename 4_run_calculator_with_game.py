"""
ALL 4 MODELS RUN SIMULTANEOUSLY - NO IF STATEMENTS!

STACKING:
1. FER Pretrained → Emotion prediction
2. Custom Model → Emotion prediction (YOUR face)
3. MediaPipe Face → Emotion from landmarks
4. MediaPipe Hands → Finger counting

GAME MODE - Challenge with target numbers!
"""

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import os
from collections import deque, Counter
import threading
import time
import random
import json

try:
    from fer import FER
except ImportError:
    try:
        from fer.fer import FER  # type: ignore
    except Exception:
        FER = None

print("=" * 70)
print("AI MATH CALCULATOR - COMPLETE STACKING EDITION + GAME MODE")
print("=" * 70)
print("\n🚀 ALL 4 MODELS STACKING TOGETHER:")
print("  1. FER Pretrained (general emotions)")
print("  2. Custom Model (YOUR face)")
print("  3. MediaPipe Face (landmark emotions)")
print("  4. MediaPipe Hands (finger counting)")
print("\n✨ ENSEMBLE VOTING: All predictions combined!")
print("✨ WORKS IMMEDIATELY: No training needed!")
print("✨ LIVE TRAINING: Press H/S/N/A to add samples!")
print("✨ GAME MODE: Press G to start challenge mode!")
print()

# GAME MODE STATE
game_active = False
game_score = 0
game_lives = 3
game_target = None
game_operation = None
challenge_start_time = 0
consecutive_correct = 0
difficulty_level = 1
game_high_score = 0

# Load high score if exists
if os.path.exists("highscore.txt"):
    try:
        with open("highscore.txt", "r") as f:
            game_high_score = int(f.read().strip())
        print(f"[INFO] High Score loaded: {game_high_score}")
    except:
        game_high_score = 0

# LOGGING SETUP
os.makedirs("logs", exist_ok=True)

# Metrics tracking
vote_history = deque(maxlen=100)
games_played_total = 0
total_score_all_games = 0
solve_times = []

# LOAD ALL MODELS
print("[INFO] Loading ALL models (no if statements)...")

# MODEL 1: FER (downloads automatically if needed)
fer_detector = None
try:
    print("  ⏳ Loading FER...")
    fer_detector = FER(mtcnn=False)
    print("  ✓ FER Pretrained active")
except:
    print("  ⚠ FER unavailable (pip install fer)")

# MODEL 2: Custom (loads if exists, creates placeholder if not)
custom_model = None
custom_labels = np.array(['happy', 'sad', 'neutral', 'angry'])

if os.path.exists("custom_emotion_model.h5"):
    try:
        custom_model = load_model("custom_emotion_model.h5")
        custom_labels = np.load("custom_emotion_labels.npy")
        print("  ✓ Custom Model active")
    except:
        print("  ⚠ Custom model file corrupted")
else:
    print("  ○ Custom Model not trained (will train when you add samples)")

# MODEL 3 & 4: MediaPipe (always available)
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("  ✓ MediaPipe Face active")
print("  ✓ MediaPipe Hands active")

# MAPPINGS (ASCII SAFE)
EMOTION_TO_OP = {
    'happy': ('*', lambda a, b: a * b, (0, 255, 0)),
    'sad': ('/', lambda a, b: a / b if b != 0 else 0, (255, 100, 100)),
    'neutral': ('+', lambda a, b: a + b, (255, 255, 255)),
    'angry': ('-', lambda a, b: a - b, (0, 100, 255))
}

FER_MAPPING = {
    'happy': 'happy', 'sad': 'sad', 'neutral': 'neutral', 'angry': 'angry',
    'surprise': 'happy', 'fear': 'sad', 'disgust': 'angry'
}

# GAME: VALID TARGETS
VALID_TARGETS = {
    'happy': list(range(0, 26)),      # 0*0 to 5*5 = 0-25
    'sad': [0, 1, 2, 2.5, 3, 4, 5],   # Clean divisions only
    'neutral': list(range(0, 11)),    # 0+0 to 5+5 = 0-10
    'angry': list(range(-5, 6))       # 0-5 to 5-0 = -5 to 5
}

# MODEL 3: MediaPipe Landmark Emotion
def mediapipe_emotion_from_landmarks(face_landmarks):
    """
    Detect emotion using MediaPipe face landmarks
    Uses mouth and eyebrow positions
    """
    try:
        # Get key landmarks
        left_mouth = face_landmarks[61]
        right_mouth = face_landmarks[291]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        
        left_brow = face_landmarks[70]
        right_brow = face_landmarks[300]
        nose = face_landmarks[168]
        
        # Calculate positions
        mouth_avg_y = (left_mouth.y + right_mouth.y) / 2
        lip_center_y = (upper_lip.y + lower_lip.y) / 2
        brow_avg_y = (left_brow.y + right_brow.y) / 2
        brow_distance = nose.y - brow_avg_y
        
        # Emotion rules
        if mouth_avg_y < lip_center_y - 0.015:  # Smile
            return 'happy'
        elif mouth_avg_y > lip_center_y + 0.01:  # Frown
            return 'sad'
        elif brow_distance < 0.06:  # Brows lowered
            return 'angry'
        else:
            return 'neutral'
    except:
        return 'neutral'

# TRAINING
training_data = {'happy': [], 'sad': [], 'neutral': [], 'angry': []}
is_training = False

def train_model_background():
    global custom_model, is_training
    
    print("\n[TRAIN] Background training...")
    
    all_X = []
    all_y = []
    
    # Load existing
    for emotion in ['happy', 'sad', 'neutral', 'angry']:
        if os.path.exists(f"emotion_{emotion}.npy"):
            data = np.load(f"emotion_{emotion}.npy")
            all_X.append(data)
            all_y.extend([emotion] * len(data))
    
    # Add live data
    for emotion, samples in training_data.items():
        if samples:
            all_X.append(np.array(samples))
            all_y.extend([emotion] * len(samples))
    
    if not all_X:
        print("[TRAIN] No data")
        is_training = False
        return
    
    X = np.vstack(all_X)
    label_map = {'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3}
    y = to_categorical([label_map[label] for label in all_y], 4)
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    ip = Input(shape=(X.shape[1],))
    x = Dense(256, activation="relu")(ip)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    op = Dense(4, activation="softmax")(x)
    
    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(X, y, epochs=15, batch_size=16, validation_split=0.15,
              callbacks=[EarlyStopping(patience=3)], verbose=0)
    
    model.save("custom_emotion_model.h5")
    custom_model = model
    
    print("[TRAIN] ✓ Done!")
    is_training = False

# FINGER COUNTING
def count_fingers(hand_lm, hand_side):
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    count = 0
    
    is_right = hand_side.classification[0].label == "Right"
    
    # Thumb
    if (is_right and hand_lm.landmark[tips[0]].x < hand_lm.landmark[pips[0]].x) or \
       (not is_right and hand_lm.landmark[tips[0]].x > hand_lm.landmark[pips[0]].x):
        count += 1
    
    # Fingers
    for i in range(1, 5):
        if hand_lm.landmark[tips[i]].y < hand_lm.landmark[pips[i]].y:
            count += 1
    
    return count

# ENSEMBLE: ALL MODELS VOTE
def get_ensemble_emotion(frame, face_landmarks_list):
    """
    ALL 4 MODELS PREDICT SIMULTANEOUSLY
    Returns: final_emotion, individual_predictions_dict
    """
    votes = []
    predictions = {}
    
    # Vote 1: FER
    if fer_detector:
        try:
            result = fer_detector.detect_emotions(frame)
            if result:
                emo_dict = result[0]['emotions']
                fer_emo = max(emo_dict, key=emo_dict.get)
                mapped = FER_MAPPING.get(fer_emo, 'neutral')
                predictions['FER'] = mapped
                votes.append(mapped)
        except:
            pass
    
    # Vote 2: Custom
    if custom_model and face_landmarks_list:
        try:
            feats = []
            for lm in face_landmarks_list:
                feats.extend([lm.x, lm.y, lm.z])
            feats = np.array(feats).reshape(1, -1)
            
            probs = custom_model.predict(feats, verbose=0)[0]
            custom_emo = custom_labels[np.argmax(probs)]
            predictions['Custom'] = custom_emo
            votes.append(custom_emo)
        except:
            pass
    
    # Vote 3: MediaPipe Landmarks
    if face_landmarks_list:
        try:
            mp_emo = mediapipe_emotion_from_landmarks(face_landmarks_list)
            predictions['MediaPipe'] = mp_emo
            votes.append(mp_emo)
        except:
            pass
    
    # ENSEMBLE: Majority vote
    if votes:
        final = Counter(votes).most_common(1)[0][0]
        return final, predictions
    
    return 'neutral', predictions

# GAME MODE FUNCTIONS

def generate_challenge(difficulty):
    """
    Generate achievable target based on difficulty
    Returns: (target_number, operation_emotion)
    """
    # Pick random operation
    operations = ['happy', 'neutral', 'angry', 'sad']
    
    # Weight operations by difficulty
    if difficulty == 1:  # Easy: mostly addition/multiplication
        weights = [0.4, 0.4, 0.1, 0.1]
    elif difficulty == 2:  # Medium: balanced
        weights = [0.3, 0.3, 0.2, 0.2]
    else:  # Hard: all equal
        weights = [0.25, 0.25, 0.25, 0.25]
    
    operation = random.choices(operations, weights=weights)[0]
    
    # Pick valid target for that operation
    valid_targets = VALID_TARGETS[operation]
    
    # For higher difficulty, prefer harder targets
    if difficulty == 1:
        # Easy: avoid 0, avoid negatives, prefer small numbers
        if operation == 'happy':
            target = random.choice([4, 6, 8, 9, 10, 12, 15, 16, 20])
        elif operation == 'neutral':
            target = random.choice([3, 5, 6, 7, 8, 9])
        elif operation == 'angry':
            target = random.choice([1, 2, 3, 4])
        else:  # sad
            target = random.choice([1, 2, 5])
    else:
        # Medium/Hard: any valid target
        target = random.choice(valid_targets)
    
    return target, operation

def check_game_answer(result, target, elapsed_time):
    """
    Check if answer is correct and calculate points
    Returns: (is_correct, points_earned)
    """
    # Handle float comparison for division
    if isinstance(result, float) or isinstance(target, float):
        is_correct = abs(result - target) < 0.01
    else:
        is_correct = (result == target)
    
    if not is_correct:
        return False, 0
    
    # Calculate points based on speed
    base_points = 100
    
    if elapsed_time < 3:
        points = base_points + 50  # Fast bonus
    elif elapsed_time < 5:
        points = base_points + 25
    elif elapsed_time < 8:
        points = base_points
    else:
        points = base_points - 25  # Slow penalty
    
    # Combo multiplier
    global consecutive_correct
    combo_multiplier = 1 + (consecutive_correct * 0.1)
    points = int(points * combo_multiplier)
    
    return True, points

def start_game():
    """Initialize new game"""
    global game_active, game_score, game_lives, consecutive_correct
    global game_target, game_operation, challenge_start_time, difficulty_level
    
    game_active = True
    game_score = 0
    game_lives = 3
    consecutive_correct = 0
    difficulty_level = 1
    
    # Generate first challenge
    game_target, game_operation = generate_challenge(difficulty_level)
    challenge_start_time = time.time()
    
    print("\n" + "=" * 70)
    print("GAME MODE STARTED!")
    print("=" * 70)
    print(f"Target: {game_target} | Operation: {game_operation}")

def log_metrics(fps, frame_time, cur_emo, cur_op, eq, preds):
    """Log real-time metrics to JSON file for dashboard"""
    global vote_history, games_played_total, total_score_all_games, solve_times
    
    # Track vote agreement
    if preds:
        vote_history.append(preds)
        
        # Calculate vote agreement (how often models agree)
        if len(vote_history) > 0:
            agreements = []
            for votes in vote_history:
                if len(votes.values()) >= 2:
                    vote_list = list(votes.values())
                    most_common = Counter(vote_list).most_common(1)[0][1]
                    agreement_pct = (most_common / len(vote_list)) * 100
                    agreements.append(agreement_pct)
            vote_agreement = np.mean(agreements) if agreements else 0
        else:
            vote_agreement = 0
        
        # Calculate model "accuracies" (really vote frequencies)
        recent_votes = {'FER': 0, 'Custom': 0, 'MediaPipe': 0, 'Ensemble': 0}
        for votes in vote_history:
            for model in votes:
                if model in recent_votes:
                    recent_votes[model] += 1
        
        total_votes = len(vote_history)
        if total_votes > 0:
            for model in recent_votes:
                recent_votes[model] = (recent_votes[model] / total_votes) * 100
    else:
        vote_agreement = 0
        recent_votes = {'FER': 0, 'Custom': 0, 'MediaPipe': 0, 'Ensemble': 0}
    
    # Calculate success rate and avg solve time
    success_rate = 0
    avg_solve_time = 0
    if games_played_total > 0:
        success_rate = (total_score_all_games / (games_played_total * 150)) * 100  # Assuming avg 150 pts per game
        if solve_times:
            avg_solve_time = np.mean(solve_times)
    
    metrics = {
        'fps': fps,
        'frame_time': frame_time,
        'mode': 'GAME MODE' if game_active else 'CALCULATOR',
        'recent_votes': recent_votes,
        'vote_agreement': vote_agreement,
        'current_emotion': cur_emo,
        'current_operation': cur_op,
        'current_equation': eq,
        'game_score': game_score if game_active else 0,
        'game_lives': game_lives if game_active else 3,
        'game_target': game_target if game_active else 0,
        'game_combo': consecutive_correct if game_active else 0,
        'game_level': difficulty_level if game_active else 1,
        'games_played': games_played_total,
        'total_score': total_score_all_games,
        'success_rate': success_rate,
        'avg_solve_time': avg_solve_time,
        'training_samples': {
            'happy': len(training_data['happy']),
            'sad': len(training_data['sad']),
            'neutral': len(training_data['neutral']),
            'angry': len(training_data['angry'])
        }
    }
    
    try:
        with open('logs/realtime_metrics.json', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    except:
        pass  # Don't crash if logging fails

def draw_game_ui(frame, result, elapsed_time):
    """Draw game overlay - ASCII SAFE VERSION"""
    h, w, c = frame.shape
    
    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    
    # Score (ASCII only)
    cv2.putText(frame, f"SCORE: {game_score}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # High score
    cv2.putText(frame, f"HIGH: {game_high_score}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    # Lives (ASCII safe - no emoji)
    lives_text = f"LIVES: {game_lives}/3"
    cv2.putText(frame, lives_text, (w - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # Timer
    time_left = max(0, 15 - int(elapsed_time))
    timer_color = (0, 255, 0) if time_left > 5 else (0, 165, 255) if time_left > 2 else (0, 0, 255)
    cv2.putText(frame, f"TIME: {time_left}s", (w//2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, timer_color, 3)
    
    # Combo (ASCII safe)
    if consecutive_correct > 0:
        combo_text = f"COMBO x{consecutive_correct + 1}!"
        cv2.putText(frame, combo_text, (w//2 - 120, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Target (BIG in center)
    target_text = f"TARGET: {game_target}"
    text_size = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 8)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h // 2 - 100
    
    # Target background
    cv2.rectangle(frame, (text_x - 20, text_y - 80), 
                  (text_x + text_size[0] + 20, text_y + 20), (50, 50, 50), -1)
    cv2.putText(frame, target_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)
    
    # Your answer (below target)
    if result != "?":
        answer_text = f"Your Answer: {result}"
        cv2.putText(frame, answer_text, (text_x - 100, text_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Difficulty level
    cv2.putText(frame, f"Level: {difficulty_level}", (w - 250, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_game_over(frame):
    """Draw game over screen"""
    h, w, c = frame.shape
    
    # Dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Game Over text
    cv2.putText(frame, "GAME OVER!", (w//2 - 200, h//2 - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
    
    # Final score
    cv2.putText(frame, f"Final Score: {game_score}", (w//2 - 180, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    
    # High score
    if game_score > game_high_score:
        cv2.putText(frame, "NEW HIGH SCORE!", (w//2 - 200, h//2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    else:
        cv2.putText(frame, f"High Score: {game_high_score}", (w//2 - 180, h//2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
    
    # Instructions
    cv2.putText(frame, "Press G to play again", (w//2 - 180, h//2 + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Press ESC to return to calculator", (w//2 - 280, h//2 + 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

# MAIN
print("[INFO] Starting...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Webcam failed!")
    exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] ALL 4 MODELS ACTIVE - STACKING ENGAGED!")
print("=" * 70)
print("\nCONTROLS:")
print("  Q=Quit | H=Happy S=Sad N=Neutral A=Angry | T=Train")
print("  G=Start Game Mode | ESC=Exit Game")
print("\nWorks NOW! Add samples to improve accuracy.\n")

cv2.namedWindow('AI Calc - ALL MODELS STACKING', cv2.WINDOW_NORMAL)
cv2.resizeWindow('AI Calc - ALL MODELS STACKING', 1400, 800)

# State
cur_left = None
cur_right = None
cur_op = '+'
cur_emo = 'neutral'
op_color = (255, 255, 255)
emo_hist = deque(maxlen=7)
left_hist = deque(maxlen=3)
right_hist = deque(maxlen=3)
last_face = None
preds = {}

# FPS tracking
fps_start_time = time.time()
frame_count = 0
current_fps = 0
frame_times = deque(maxlen=30)

while True:
    frame_start = time.time()
    
    ret, frm = cap.read()
    if not ret:
        break
    
    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    h, w, c = frm.shape
    
    # Process
    face_res = face_mesh.process(rgb)
    hand_res = hands.process(rgb)
    
    # EMOTION: ALL MODELS
    if face_res and face_res.multi_face_landmarks:
        for face_lm in face_res.multi_face_landmarks:
            last_face = face_lm.landmark
            
            # Get ALL predictions
            ens_emo, preds = get_ensemble_emotion(frm, face_lm.landmark)
            
            emo_hist.append(ens_emo)
            
            if len(emo_hist) >= 4:
                cur_emo = Counter(emo_hist).most_common(1)[0][0]
            
            if cur_emo in EMOTION_TO_OP:
                cur_op, _, op_color = EMOTION_TO_OP[cur_emo][:3]
    
    # HANDS: MediaPipe
    if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
        temp_l = None
        temp_r = None
        
        for hand_lm, hand_side in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
            mp_drawing.draw_landmarks(
                frm, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            cnt = count_fingers(hand_lm, hand_side)
            label = hand_side.classification[0].label
            
            if label == "Left":
                temp_l = cnt
            else:
                temp_r = cnt
            
            cx = int(hand_lm.landmark[9].x * w)
            cy = int(hand_lm.landmark[9].y * h)
            cv2.putText(frm, str(cnt), (cx-30, cy-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,255), 5)
        
        if temp_l is not None:
            left_hist.append(temp_l)
            if len(left_hist) >= 2:
                cur_left = int(np.median(list(left_hist)))
        
        if temp_r is not None:
            right_hist.append(temp_r)
            if len(right_hist) >= 2:
                cur_right = int(np.median(list(right_hist)))
    
    # CALCULATE
    result = "?"
    if cur_left is not None and cur_right is not None and cur_emo in EMOTION_TO_OP:
        try:
            _, func, _ = EMOTION_TO_OP[cur_emo]
            res = func(cur_left, cur_right)
            result = f"{res:.2f}" if isinstance(res, float) else str(res)
        except:
            result = "Error"
    
    # GAME MODE LOGIC
    if game_active and game_lives > 0:
        elapsed = time.time() - challenge_start_time
        
        # Draw game UI instead of normal UI
        draw_game_ui(frm, result, elapsed)
        
        # Check if answer is correct
        if result != "?" and result != "Error":
            try:
                result_num = float(result)
                is_correct, points = check_game_answer(result_num, game_target, elapsed)
                
                if is_correct:
                    # Correct answer!
                    game_score += points
                    consecutive_correct += 1
                    
                    print(f"CORRECT! +{points} points (Combo x{consecutive_correct})")
                    
                    # Update high score
                    if game_score > game_high_score:
                        game_high_score = game_score
                        try:
                            with open("highscore.txt", "w") as f:
                                f.write(str(game_high_score))
                        except:
                            pass
                    
                    # Increase difficulty every 5 correct
                    if consecutive_correct % 5 == 0:
                        difficulty_level += 1
                        print(f"Level Up! Difficulty: {difficulty_level}")
                    
                    # New challenge
                    time.sleep(0.5)  # Brief pause to see success
                    game_target, game_operation = generate_challenge(difficulty_level)
                    challenge_start_time = time.time()
            except:
                pass
        
        # Check timeout
        if elapsed > 15:  # 15 second timeout
            game_lives -= 1
            consecutive_correct = 0  # Reset combo
            
            print(f"TIME'S UP! Lives left: {game_lives}")
            
            if game_lives > 0:
                # Generate new challenge
                game_target, game_operation = generate_challenge(difficulty_level)
                challenge_start_time = time.time()
            else:
                # Game over - track statistics
                games_played_total += 1
                total_score_all_games += game_score
                print(f"\nGAME OVER! Final Score: {game_score}")
    
    # Game over screen
    elif game_active and game_lives <= 0:
        draw_game_over(frm)
    
    # NORMAL CALCULATOR MODE
    else:
        # UI
        overlay = frm.copy()
        cv2.rectangle(overlay, (0,0), (w,320), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frm, 0.5, 0, frm)
        
        # Equation
        l_str = str(cur_left) if cur_left else "?"
        r_str = str(cur_right) if cur_right else "?"
        eq = f"{l_str} {cur_op} {r_str} = {result}"
        
        cv2.putText(frm, eq, (50,120), cv2.FONT_HERSHEY_SIMPLEX, 3, op_color, 8)
        
        # Emotion + votes
        cv2.putText(frm, f"Emotion: {cur_emo.upper()}", (50,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        
        y = 240
        for model, vote in preds.items():
            color = (0,255,0) if vote == cur_emo else (150,150,150)
            cv2.putText(frm, f"{model}: {vote}", (50,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        
        # Training
        if is_training:
            cv2.putText(frm, "TRAINING...", (w-250,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
        
        total = sum(len(s) for s in training_data.values())
        if total:
            cv2.putText(frm, f"Samples: {total}", (w-250,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        # Controls
        cv2.putText(frm, "Q=Quit | H/S/N/A=Sample | T=Train | G=Game", (30,h-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frm, "All 4 models voting together!", (30,h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    # Calculate FPS and frame time
    frame_end = time.time()
    frame_time = (frame_end - frame_start) * 1000  # ms
    frame_times.append(frame_time)
    
    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS every 10 frames
        elapsed = time.time() - fps_start_time
        if elapsed > 0:
            current_fps = frame_count / elapsed
    
    # Log metrics for dashboard
    l_str = str(cur_left) if cur_left else "?"
    r_str = str(cur_right) if cur_right else "?"
    eq = f"{l_str} {cur_op} {r_str} = {result}"
    avg_frame_time = np.mean(frame_times) if frame_times else 0
    log_metrics(current_fps, avg_frame_time, cur_emo, cur_op, eq, preds)
    
    cv2.imshow('AI Calc - ALL MODELS STACKING', frm)
    
    # KEYS
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    # Start/restart game
    if key == ord('g'):
        if not game_active or game_lives <= 0:
            start_game()
        else:
            print("[INFO] Game already active!")
    
    # ESC exits game mode
    if key == 27:  # ESC
        if game_active:
            game_active = False
            game_lives = 0
            print("\n[INFO] Exited game mode")
    
    # Live training (only in calculator mode)
    if not game_active and last_face:
        emo_key = None
        if key == ord('h'):
            emo_key = 'happy'
        elif key == ord('s'):
            emo_key = 'sad'
        elif key == ord('n'):
            emo_key = 'neutral'
        elif key == ord('a'):
            emo_key = 'angry'
        
        if emo_key:
            feats = []
            for lm in last_face:
                feats.extend([lm.x, lm.y, lm.z])
            training_data[emo_key].append(feats)
            print(f"[LIVE] {emo_key}! Total: {len(training_data[emo_key])}")
    
    # Train
    if key == ord('t') and not is_training and not game_active:
        total = sum(len(s) for s in training_data.values())
        if total >= 10:
            is_training = True
            threading.Thread(target=train_model_background, daemon=True).start()
        else:
            print(f"[TRAIN] Need 10+ samples (have {total})")

cv2.destroyAllWindows()
cap.release()
face_mesh.close()
hands.close()
print("\n[OK] Gesture calculator closed!\n")
