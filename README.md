# GESTURE + EMOTION CALCULATOR

   # Facial Operator


   ### What you'll find in our repo:

   - Scripts to collect data, test hands, train, and run the demo: `1_collect_emotions.py`, `2_test_hands.py`, `3_train_emotions.py`, `4_run_calculator_with_game.py`, and helpers.
   - Documentation files: `README_FINAL.md`.
   - A `requirements.txt` that lists the Python dependencies.

   #### How to start

   1. Create and activate a virtual environment (requires Python 3.11.14):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   3. Run the game:

   ```bash
   python3 4_run_calculator.py
   ```

   This will open a window showing your camera feed (Ensure system settings on your laptop are configured to allow this to happen). Use your left hand to enter the first number, make a face to use an operator, and use your right hand to enter the second number!

   Controls

   - `Q` or 'esc' : Quit
   - `H`, `S`, `N`, `A`: Add labeled samples (happy, sad, neutral, angry)
   - `T`: Train the classifier on collected samples
   - `G`: Start the game

   #### Game Mode

   Press `G` to play the math challenge game! You get 15 seconds to solve each problem and have 3 lives.

   How do the points work? 
   - Every correct answer gives you 100 points
   - Answer fast for bonus points:
     - Under 3 seconds: Get 50 extra points (150 total)
     - Under 5 seconds: Get 25 extra points (125 total)
     - Under 8 seconds: Get the normal 100 points
     - Over 8 seconds: Lose 25 points (only 75 total)
   - Build a combo streak!! Each correct answer in a row makes your points worth more:
     - 1st correct answer: Normal points
     - 2nd correct in a row: 10% more points
     - 3rd correct in a row: 20% more points
     - Keep going to earn even more!
   - The game gets harder every 5 correct answers
   - If time runs out or you get it wrong, you lose a life and your combo resets D:
   - Game over when you run out of lives.

   #### Notes and tips: 
   - Good lighting and a steady camera make the game more reliable.
   - Make exaggerated facial expressions! Have some fun with it.
   - Collect several samples per expression before training to improve the learned model. 
