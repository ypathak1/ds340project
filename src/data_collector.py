"""Mouth-region data collector using MediaPipe Face Mesh.

Run and press keys 1..N to save samples for corresponding classes. Press 'q' or ESC to quit.
"""
import os
import time
from pathlib import Path
import cv2
import argparse

try:
    import mediapipe as mp
except Exception:
    mp = None

from .utils import crop_mouth_from_face_mesh


def ensure_dirs(base_dir, classes):
    base = Path(base_dir)
    for c in classes:
        (base / 'train' / c).mkdir(parents=True, exist_ok=True)
        (base / 'val' / c).mkdir(parents=True, exist_ok=True)


def run_collector(output_dir='data/collected', classes=None, sample_prefix=None):
    if mp is None:
        print('mediapipe not installed. Install package first.')
        return

    if classes is None:
        classes = ['neutral', 'frown', 'smile', 'tongue', 'open']

    ensure_dirs(output_dir, classes)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    mp_face = mp.solutions.face_mesh

    print('Data collector started.')
    print('Press keys 1..{} to save sample for classes:'.format(len(classes)))
    for i, c in enumerate(classes, start=1):
        print(f'  {i}: {c}')
    print("Press 'v' to toggle saving to val instead of train. Press 'q' or ESC to quit.")

    save_to_val = False
    counter = {c: 0 for c in classes}

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                crop = crop_mouth_from_face_mesh(frame_rgb, landmarks)
                preview = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                ph, pw = 224, 224
                preview = cv2.resize(preview, (pw, ph))
                frame[0:ph, 0:pw] = preview

            cv2.putText(frame, f'Saving to: {"val" if save_to_val else "train"}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Data Collector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('v'):
                save_to_val = not save_to_val

            # keys 49..57 correspond to '1'..'9'
            if 49 <= key <= 57:
                idx = key - 49  # 0-based
                if idx < len(classes):
                    cls = classes[idx]
                    subset = 'val' if save_to_val else 'train'
                    out_dir = Path(output_dir) / subset / cls
                    # if no face detected, skip
                    if not results.multi_face_landmarks:
                        print('No face detected; sample not saved.')
                        continue
                    crop = crop_mouth_from_face_mesh(frame_rgb, results.multi_face_landmarks[0].landmark)
                    timestamp = int(time.time() * 1000)
                    fname = out_dir / f'{cls}_{timestamp}.png'
                    # save as RGB->BGR via cv2
                    cv2.imwrite(str(fname), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    counter[cls] += 1
                    print(f'Saved {fname} (count {counter[cls]})')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/collected')
    parser.add_argument('--classes', default='neutral,frown,smile,tongue,open')
    args = parser.parse_args()
    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    run_collector(output_dir=args.out, classes=classes)
