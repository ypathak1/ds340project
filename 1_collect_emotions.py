"""
STEP 1: Collect Emotion Data
1. Live webcam collection
2. Load images from folder
3. Use both methods
"""

import mediapipe as mp 
import numpy as np 
import cv2
import os
from pathlib import Path

print("=" * 70)
print("STEP 1: EMOTION DATA COLLECTION - HYBRID MODE")
print("=" * 70)
print("\nCollect emotion training data in 3 ways:")
print("  1. LIVE: Use webcam (real-time)")
print("  2. MANUAL: Load images from folder")
print("  3. BOTH: Combine webcam + manual images")
print("\nEmotions to collect:")
print("  - happy")
print("  - sad")
print("  - neutral")
print("  - angry")
print()

# Get emotion name
valid_emotions = ['happy', 'sad', 'neutral', 'angry']
print(f"Valid emotions: {', '.join(valid_emotions)}")
emotion = input("Enter emotion name: ").strip().lower()

if emotion not in valid_emotions:
    print(f"Error: Must be one of: {', '.join(valid_emotions)}")
    exit(1)

# Ask collection method
print("\n" + "=" * 70)
print("COLLECTION METHOD:")
print("=" * 70)
print("1. WEBCAM - Collect from live webcam (recommended for personalization)")
print("2. FOLDER - Load images from a folder")
print("3. BOTH - Use webcam first, then add folder images")
print()

method = input("Choose method (1/2/3): ").strip()

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

X = []
data_size = 0

# METHOD 1
if method in ['1', '3']:
    print("\n" + "=" * 70)
    print("WEBCAM COLLECTION")
    print("=" * 70)
    print(f"[INFO] Collecting {emotion} from webcam")
    print("[INFO] Make the expression and hold it steady")
    print("[INFO] Press SPACE to capture, ESC when done")
    print()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        exit(1)
    
    webcam_count = 0
    
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        
        frm = cv2.flip(frm, 1)
        rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=frm,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
        
        # Display info
        cv2.putText(frm, f"Emotion: {emotion.upper()}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frm, f"Captured: {webcam_count}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frm, "SPACE=Capture  ESC=Done", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Webcam Collection - SPACE to capture", frm)
        
        key = cv2.waitKey(1)
        
        # SPACE to capture
        if key == 32:  # SPACE
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lst = []
                    for landmark in face_landmarks.landmark:
                        lst.append(landmark.x)
                        lst.append(landmark.y)
                        lst.append(landmark.z)
                    X.append(lst)
                    webcam_count += 1
                    data_size += 1
                    print(f"[CAPTURED] Sample #{webcam_count}")
        
        # ESC to finish
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    print(f"\n[OK] Webcam collection complete: {webcam_count} samples")

# METHOD 2
if method in ['2', '3']:
    print("\n" + "=" * 70)
    print("FOLDER COLLECTION")
    print("=" * 70)
    print("Load images from a folder on your computer.")
    print(f"Put images of '{emotion}' expression in a folder.")
    print()
    
    folder_path = input("Enter folder path (or press Enter to skip): ").strip()
    
    if folder_path and os.path.exists(folder_path):
        print(f"\n[INFO] Loading images from: {folder_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        folder_count = 0
        
        for img_file in Path(folder_path).iterdir():
            if img_file.suffix.lower() in image_extensions:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_img)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lst = []
                        for landmark in face_landmarks.landmark:
                            lst.append(landmark.x)
                            lst.append(landmark.y)
                            lst.append(landmark.z)
                        X.append(lst)
                        folder_count += 1
                        data_size += 1
                    print(f"[LOADED] {img_file.name} -> Sample #{folder_count}")
                else:
                    print(f"[SKIP] {img_file.name} - No face detected")
        
        print(f"\n[OK] Folder collection complete: {folder_count} samples")
    elif folder_path:
        print(f"[WARNING] Folder not found: {folder_path}")

face_mesh.close()

# SAVE DATA
if data_size > 0:
    np.save(f"emotion_{emotion}.npy", np.array(X))
    print("\n" + "=" * 70)
    print("[SUCCESS] Data saved!")
    print("=" * 70)
    print(f"[FILE] emotion_{emotion}.npy")
    print(f"[SAMPLES] {data_size} total")
    print(f"[SHAPE] {np.array(X).shape}")
    print()
else:
    print("\n[WARNING] No data collected!")
