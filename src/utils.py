import cv2
import numpy as np
from collections import deque, Counter


def crop_mouth_from_face_mesh(image, face_landmarks, output_size=(224, 224)):
    """Given an RGB image and MediaPipe face landmarks, crop a mouth-centered bounding box.

    face_landmarks: iterable of (x, y) landmark normalized coordinates (0..1) as from MediaPipe.
    Returns the resized crop (RGB).
    """
    h, w = image.shape[:2]
    # mouth region indices from MediaPipe FaceMesh (approx): 61..88 contain lips
    mouth_idx = list(range(61, 89))
    pts = []
    for i, lm in enumerate(face_landmarks):
        if i in mouth_idx:
            pts.append((int(lm.x * w), int(lm.y * h)))

    if len(pts) == 0:
        # fallback: center crop
        cy, cx = h // 2, w // 2
        half = min(h, w) // 6
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        crop = image[y1:y2, x1:x2]
    else:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0, min(xs) - 8), min(w, max(xs) + 8)
        y1, y2 = max(0, min(ys) - 8), min(h, max(ys) + 8)
        crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        # as last resort, return center crop
        ch, cw = h // 6, w // 6
        cy, cx = h // 2, w // 2
        crop = image[cy - ch:cy + ch, cx - cw:cx + cw]

    crop = cv2.resize(crop, output_size)
    return crop


class MajorityVoteSmoother:
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)

    def add(self, label):
        self.queue.append(label)

    def get(self):
        if len(self.queue) == 0:
            return None
        cnt = Counter(self.queue)
        return cnt.most_common(1)[0][0]
