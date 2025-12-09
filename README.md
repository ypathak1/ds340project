# GESTURE + EMOTION CALCULATOR

   # Facial Operator


   What you'll find here

   - Scripts to collect data, test hands, train, and run the demo: `1_collect_emotions.py`, `2_test_hands.py`, `3_train_emotions.py`, `4_run_calculator_with_game.py`, and helpers.
   - Documentation files: `README_FINAL.md`.
   - A `requirements.txt` that lists the Python dependencies.

   How to start

   1. Create and activate a virtual environment (requires Python 3.11.14):

   ```bash
   python3.11 -m venv .venv
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

   The demo opens a window showing the camera feed. Use your left hand to enter the first number, make a face to use an operator, and use your right hand to enter the second number!

   Controls

   - `Q` or 'esc' : Quit
   - `H`, `S`, `N`, `A`: Add labeled samples (happy, sad, neutral, angry)
   - `T`: Train the classifier on collected samples

   Notes and tips

   - Good lighting and a steady camera make the demo more reliable.
   - Make exaggerated facial expressions! Have some fun with it.
   - Collect several samples per expression before training to improve the learned model. 
