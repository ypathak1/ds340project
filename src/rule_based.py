import numpy as np


def mouth_open_ratio(landmarks, image_size=None):
    """Compute a simple vertical/horizontal mouth ratio using landmarks.

    landmarks: MediaPipe face landmarks sequence where each lm has x,y (normalized).
    Returns vertical/horizontal ratio.
    """
    # Points: upper lip and lower lip — use approximate indices
    # upper: 13, lower: 14 (these indices may vary; MediaPipe has many landmark indices)
    try:
        upper = landmarks[13]
        lower = landmarks[14]
    except Exception:
        return 0.0

    vy = abs(upper.y - lower.y)

    # width: use left and right mouth corners
    try:
        left = landmarks[61]
        right = landmarks[291]
        vx = abs(left.x - right.x)
    except Exception:
        vx = 1e-6

    if vx == 0:
        return 0.0
    return vy / vx


def rule_based_operator(landmarks):
    """Return operator label based on simple geometric heuristics.

    Labels: '+': neutral, '-': frown (small open ratio but downturned?),
    'x': tongue-out (we detect large vertical protrusion near center), '/': smile (wide width)
    This is a very approximate heuristic for baseline comparison.
    """
    ratio = mouth_open_ratio(landmarks)
    # thresholds are heuristic and will require tuning
    if ratio < 0.02:
        return '+'  # closed / neutral
    if ratio > 0.12:
        # large open mouth -> could be tongue-out or surprise; call 'x'
        return 'x'
    # width-based heuristic for smile vs frown
    try:
        left = landmarks[61]
        right = landmarks[291]
        mouth_width = abs(left.x - right.x)
    except Exception:
        mouth_width = 0.3

    if mouth_width > 0.45:
        return '÷'  # smile/wide
    return '−'  # frown/other
