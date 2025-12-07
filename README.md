# GESTURE + EMOTION CALCULATOR

```
   # Facial Operator — demo and data collector

   This repo contains a compact demo that uses MediaPipe for face and hand landmarks and a small, easy-to-read pipeline to recognize simple expressions and finger counts. The demo is intended to be understandable and easy to run on a laptop camera.

   What you'll find here

   - Small scripts to collect data, test hands, train, and run the demo: `1_collect_emotions.py`, `2_test_hands.py`, `3_train_emotions.py`, `4_run_calculator.py`, and helpers.
   - Documentation files: `PROJECT_DOCUMENTATION.md`, `README_FINAL.md`.
   - A `requirements.txt` that lists the Python dependencies.

   How to start

   1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   3. Run the demo:

   ```bash
   python3 4_run_calculator.py
   ```

   The demo opens a window showing the camera feed. Use your left hand to enter the first number (0–5 fingers), make a face to select an operator, and use your right hand to enter the second number.

   Controls

   - `Q`: quit
   - `H`, `S`, `N`, `A`: add labeled samples (happy, sad, neutral, angry)
   - `T`: train the classifier on collected samples

   Notes and tips

   - Good lighting and a steady camera make the demo more reliable.
   - The operator detector is intentionally rule-based so you can see how it works and change it easily.
   - Collect several samples per expression before training to improve the learned model.