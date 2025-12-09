"""
STEP 2: Test MediaPipe Hand Counting
MediaPipe automatically counts fingers
"""

import mediapipe as mp
import cv2
import time

print("=" * 70)
print("STEP 2: MEDIAPIPE HAND COUNTING TEST")
print("=" * 70)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def count_fingers(hand_landmarks, handedness):
    """
    Count extended fingers using MediaPipe landmarks
    Returns number 0-5
    """
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [2, 6, 10, 14, 18]  # Joints below tips
    
    fingers_up = 0
    
    # Thumb, check based on hand side
    is_right_hand = handedness.classification[0].label == "Right"
    
    if is_right_hand:
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
            fingers_up += 1
    else:
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
            fingers_up += 1
    
    # Other four fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
            fingers_up += 1
    
    return fingers_up

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam!")
    exit(1)

print("[INFO] Testing MediaPipe hand tracking...")
print("[INFO] Hold up different numbers of fingers!")
print("[INFO] Try: 0 (fist), 1, 2, 3, 4, 5 (open palm)")
print("[INFO] Press ESC to finish\n")

cv2.namedWindow('MediaPipe Hand Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Hand Test', 1280, 720)

test_duration = 30  # seconds
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    h, w, c = frame.shape
    
    elapsed = int(time.time() - start_time)
    remaining = max(0, test_duration - elapsed)
    
    cv2.putText(frame, "MediaPipe Finger Counting Test", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"Time: {remaining}s  |  Press ESC to exit", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            finger_count = count_fingers(hand_landmarks, handedness)
            hand_label = handedness.classification[0].label
            
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            
            cv2.putText(frame, f"{hand_label}: {finger_count}", (cx - 70, cy - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    else:
        cv2.putText(frame, "No hands detected - show your hands!", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('MediaPipe Hand Test', frame)
    
    if cv2.waitKey(1) == 27 or remaining <= 0:
        break

cv2.destroyAllWindows()
cap.release()
hands.close()

print("\n" + "=" * 70)
print("[SUCCESS] MediaPipe test complete!")
print("=" * 70)
print("\n If finger counts appeared, MediaPipe is working!")
print("Next: Run 3_train_emotions.py\n")
