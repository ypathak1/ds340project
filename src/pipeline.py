"""Real-time pipeline script: face mesh -> crop mouth -> classify -> smoothing -> combine with hands

Run as: python -m src.pipeline
"""
import time
import argparse
import collections
import os

import cv2
import numpy as np

# Reduce verbose TensorFlow/MediaPipe logging when possible
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    import absl.logging
    absl.logging.set_verbosity('error')
except Exception:
    pass

try:
    import mediapipe as mp
except Exception:
    mp = None

# Keep pipeline lightweight: no ML model by default in this simplified demo.

from .utils import crop_mouth_from_face_mesh, MajorityVoteSmoother
from .rule_based import rule_based_operator


def count_fingers(hand_landmarks):
    """A simple finger-count heuristic using landmarks (MediaPipe Hands)."""
    # This heuristic assumes landmarks are the MediaPipe structure with normalized x,y
    # We'll count fingers by checking tip vs pip y-coordinate for fingers (thumb is special).
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    cnt = 0
    try:
        for tip_i, pip_i in zip(tips, pips):
            tip = hand_landmarks.landmark[tip_i]
            pip = hand_landmarks.landmark[pip_i]
            # for simplicity, compare y (works for up-facing camera); more robust logic can be added
            if tip.y < pip.y:
                cnt += 1
    except Exception:
        return 0
    return cnt


def run_demo(device='cpu', model_checkpoint=None, smoother_win=7):
    if mp is None:
        print('mediapipe is not installed. Install dependencies from requirements.txt')
        return

    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    # No ML model in this simplified release; operator detection uses the rule-based baseline.
    model = None
    transform = None
    smoother = MajorityVoteSmoother(window_size=smoother_win)

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:

        last_time = time.time()
        fps_deque = collections.deque(maxlen=30)

        # Simple arithmetic state: operand1, operator, operand2
        operand1 = None
        operator = None
        operand2 = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            # defaults for this frame
            operator_label = None
            smoothed = None
            ph, pw = 160, 160

            # hand digit detection (do this early so digit is available for state machine)
            digit = None
            if hand_results and getattr(hand_results, 'multi_hand_landmarks', None):
                if hand_results.multi_hand_landmarks:
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    digit = count_fingers(hand_landmarks)
                    cv2.putText(frame, f'Digit: {digit}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

            if results and getattr(results, 'multi_face_landmarks', None):
                face_landmarks = results.multi_face_landmarks[0].landmark
                # crop mouth
                mouth_crop = crop_mouth_from_face_mesh(frame_rgb, face_landmarks)

                # rule-based operator (simple, explainable baseline)
                operator_label = rule_based_operator(face_landmarks)

                smoother.add(operator_label)
                smoothed = smoother.get()

                # draw mouth crop preview
                h, w = frame.shape[:2]
                # place small preview in top-left
                preview = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR)
                ph, pw = 160, 160
                preview = cv2.resize(preview, (pw, ph))
                frame[0:ph, 0:pw] = preview

                if smoothed is not None:
                    cv2.putText(frame, f'Op: {smoothed}', (10, ph + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            # combine operator with detected digit (simple state machine)
            if digit is not None and smoothed is not None:
                # map operator symbols to functions
                try:
                    # convert digit to int if possible
                    d = int(digit)
                except Exception:
                    d = None

                if d is not None:
                    if operand1 is None:
                        operand1 = d
                        operator = smoothed
                    elif operator is not None and operand2 is None:
                        operand2 = d

                # compute if we have full expression
                result_str = ''
                if operand1 is not None and operator is not None and operand2 is not None:
                    res = None
                    if operator == '+':
                        res = operand1 + operand2
                    elif operator == '−':
                        res = operand1 - operand2
                    elif operator == 'x':
                        res = operand1 * operand2
                    elif operator == '÷':
                        res = operand1 / operand2 if operand2 != 0 else None

                    if res is not None:
                        result_str = f'{operand1} {operator} {operand2} = {res}'
                    else:
                        result_str = f'{operand1} {operator} {operand2} = ?'

                    # show result and reset
                    cv2.putText(frame, result_str, (10, ph + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                    operand1 = None
                    operator = None
                    operand2 = None

            else:
                # if no complete expression, show partial state
                state = ''
                if operand1 is not None:
                    state += str(operand1)
                if operator is not None:
                    state += f' {operator} '
                if operand2 is not None:
                    state += str(operand2)
                if state:
                    cv2.putText(frame, state, (10, ph + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 1)

            # (hand detection was handled earlier)

            # compute FPS
            now = time.time()
            fps = 1.0 / (now - last_time) if now > last_time else 0.0
            last_time = now
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow('Facial Operator Demo', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--smoother', type=int, default=7)
    args = parser.parse_args()
    run_demo(device=args.device, model_checkpoint=args.checkpoint, smoother_win=args.smoother)
