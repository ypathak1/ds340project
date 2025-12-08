# -*- coding: utf-8 -*-
"""
STEP 7: REAL-TIME TERMINAL DASHBOARD
Displays live metrics while the calculator runs
Shows model votes, accuracy, FPS, and game statistics

Usage: Run this ALONGSIDE 4_run_calculator_with_game.py
It reads from the log file that the calculator generates
"""

import time
import os
import json
from collections import deque
import sys

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def create_bar(value, max_value=100, length=20):
    """Create a progress bar string"""
    filled = int((value / max_value) * length)
    bar = '█' * filled + '░' * (length - filled)
    return bar

def print_dashboard(metrics):
    """Print the dashboard to terminal"""
    clear_screen()
    
    # Header
    print(Colors.BOLD + Colors.CYAN + "=" * 70)
    print(" " * 15 + "GESTURE CALCULATOR - LIVE DASHBOARD")
    print("=" * 70 + Colors.ENDC)
    print()
    
    # System Status
    print(Colors.BOLD + "SYSTEM STATUS" + Colors.ENDC)
    print("-" * 70)
    
    if metrics['fps'] > 25:
        fps_color = Colors.GREEN
    elif metrics['fps'] > 20:
        fps_color = Colors.YELLOW
    else:
        fps_color = Colors.RED
    
    print(f"FPS:              {fps_color}{metrics['fps']:>6.1f}{Colors.ENDC}")
    print(f"Frame Time:       {metrics['frame_time']:>6.1f} ms")
    print(f"Mode:             {Colors.CYAN}{metrics['mode']:>15}{Colors.ENDC}")
    print()
    
    # Model Votes
    print(Colors.BOLD + "MODEL PERFORMANCE (Last 100 frames)" + Colors.ENDC)
    print("-" * 70)
    
    for model, accuracy in metrics['model_accuracies'].items():
        bar = create_bar(accuracy, 100, 30)
        if accuracy >= 90:
            color = Colors.GREEN
        elif accuracy >= 80:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        print(f"{model:<12} {bar}  {color}{accuracy:>5.1f}%{Colors.ENDC}")
    
    # Vote agreement
    agreement = metrics.get('vote_agreement', 0)
    agreement_bar = create_bar(agreement, 100, 30)
    agreement_color = Colors.GREEN if agreement > 70 else Colors.YELLOW if agreement > 50 else Colors.RED
    print(f"\n{'Agreement':<12} {agreement_bar}  {agreement_color}{agreement:>5.1f}%{Colors.ENDC}")
    print()
    
    # Current State
    print(Colors.BOLD + "CURRENT STATE" + Colors.ENDC)
    print("-" * 70)
    print(f"Emotion:          {Colors.CYAN}{metrics['current_emotion'].upper():>15}{Colors.ENDC}")
    print(f"Operation:        {Colors.YELLOW}{metrics['current_operation']:>15}{Colors.ENDC}")
    print(f"Equation:         {metrics['current_equation']:>15}")
    print()
    
    # Game Mode Stats (if active)
    if metrics['mode'] == 'GAME MODE':
        print(Colors.BOLD + "GAME STATISTICS" + Colors.ENDC)
        print("-" * 70)
        print(f"Score:            {Colors.YELLOW}{metrics['game_score']:>15}{Colors.ENDC}")
        print(f"Lives:            {Colors.RED}{metrics['game_lives']:>15}{Colors.ENDC}")
        print(f"Target:           {Colors.CYAN}{metrics['game_target']:>15}{Colors.ENDC}")
        print(f"Combo:            {Colors.GREEN}x{metrics['game_combo']:>14}{Colors.ENDC}")
        print(f"Level:            {metrics['game_level']:>15}")
        print()
        
        # Game analytics
        if metrics['games_played'] > 0:
            print(Colors.BOLD + "SESSION STATS" + Colors.ENDC)
            print("-" * 70)
            print(f"Games Played:     {metrics['games_played']:>15}")
            print(f"Total Score:      {metrics['total_score']:>15}")
            print(f"Success Rate:     {Colors.GREEN}{metrics['success_rate']:>14.1f}%{Colors.ENDC}")
            print(f"Avg Solve Time:   {metrics['avg_solve_time']:>14.1f}s")
            print()
    
    # Training Data
    if metrics['training_samples'] > 0:
        print(Colors.BOLD + "TRAINING DATA" + Colors.ENDC)
        print("-" * 70)
        print(f"Total Samples:    {metrics['training_samples']:>15}")
        print(f"Happy:            {metrics['samples_happy']:>15}")
        print(f"Sad:              {metrics['samples_sad']:>15}")
        print(f"Neutral:          {metrics['samples_neutral']:>15}")
        print(f"Angry:            {metrics['samples_angry']:>15}")
        print()
    
    # Footer
    print(Colors.BOLD + Colors.CYAN + "=" * 70 + Colors.ENDC)
    print(f"{Colors.YELLOW}Press Ctrl+C to exit dashboard{Colors.ENDC}")
    print()

def main():
    """Main dashboard loop"""
    print(Colors.BOLD + Colors.CYAN)
    print("=" * 70)
    print(" " * 20 + "STARTING DASHBOARD")
    print("=" * 70)
    print(Colors.ENDC)
    print()
    print("This dashboard displays real-time metrics from the calculator.")
    print()
    print(f"{Colors.YELLOW}Instructions:{Colors.ENDC}")
    print("  1. Run this script in one terminal")
    print("  2. Run 4_run_calculator_with_game.py in another terminal")
    print("  3. The calculator will log metrics to 'logs/realtime_metrics.json'")
    print("  4. This dashboard will display them in real-time")
    print()
    print(f"{Colors.GREEN}Waiting for calculator to start...{Colors.ENDC}")
    print()
    
    # Wait for log file to exist
    while not os.path.exists('logs/realtime_metrics.json'):
        time.sleep(0.5)
    
    print(f"{Colors.GREEN}✓ Calculator detected! Starting dashboard...{Colors.ENDC}")
    time.sleep(2)
    
    # Initialize metrics
    metrics = {
        'fps': 0,
        'frame_time': 0,
        'mode': 'CALCULATOR',
        'model_accuracies': {
            'FER': 0,
            'Custom': 0,
            'MediaPipe': 0,
            'Ensemble': 0
        },
        'vote_agreement': 0,
        'current_emotion': 'neutral',
        'current_operation': '+',
        'current_equation': '? + ? = ?',
        'game_score': 0,
        'game_lives': 3,
        'game_target': 0,
        'game_combo': 0,
        'game_level': 1,
        'games_played': 0,
        'total_score': 0,
        'success_rate': 0,
        'avg_solve_time': 0,
        'training_samples': 0,
        'samples_happy': 0,
        'samples_sad': 0,
        'samples_neutral': 0,
        'samples_angry': 0
    }
    
    try:
        while True:
            # Read latest metrics from log file
            try:
                with open('logs/realtime_metrics.json', 'r') as f:
                    data = json.load(f)
                    
                    # Update metrics
                    metrics['fps'] = data.get('fps', 0)
                    metrics['frame_time'] = data.get('frame_time', 0)
                    metrics['mode'] = data.get('mode', 'CALCULATOR')
                    
                    # Model accuracies (simulated from votes)
                    votes = data.get('recent_votes', {})
                    for model in ['FER', 'Custom', 'MediaPipe', 'Ensemble']:
                        if model in votes:
                            metrics['model_accuracies'][model] = votes[model]
                    
                    metrics['vote_agreement'] = data.get('vote_agreement', 0)
                    metrics['current_emotion'] = data.get('current_emotion', 'neutral')
                    metrics['current_operation'] = data.get('current_operation', '+')
                    metrics['current_equation'] = data.get('current_equation', '? + ? = ?')
                    
                    # Game stats
                    if metrics['mode'] == 'GAME MODE':
                        metrics['game_score'] = data.get('game_score', 0)
                        metrics['game_lives'] = data.get('game_lives', 3)
                        metrics['game_target'] = data.get('game_target', 0)
                        metrics['game_combo'] = data.get('game_combo', 0)
                        metrics['game_level'] = data.get('game_level', 1)
                    
                    # Session stats
                    metrics['games_played'] = data.get('games_played', 0)
                    metrics['total_score'] = data.get('total_score', 0)
                    metrics['success_rate'] = data.get('success_rate', 0)
                    metrics['avg_solve_time'] = data.get('avg_solve_time', 0)
                    
                    # Training data
                    training = data.get('training_samples', {})
                    metrics['samples_happy'] = training.get('happy', 0)
                    metrics['samples_sad'] = training.get('sad', 0)
                    metrics['samples_neutral'] = training.get('neutral', 0)
                    metrics['samples_angry'] = training.get('angry', 0)
                    metrics['training_samples'] = sum([
                        metrics['samples_happy'],
                        metrics['samples_sad'],
                        metrics['samples_neutral'],
                        metrics['samples_angry']
                    ])
                    
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # Keep previous metrics if file doesn't exist or is invalid
            
            # Display dashboard
            print_dashboard(metrics)
            
            # Update every 0.5 seconds
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        clear_screen()
        print()
        print(Colors.BOLD + Colors.CYAN + "=" * 70)
        print(" " * 20 + "DASHBOARD STOPPED")
        print("=" * 70 + Colors.ENDC)
        print()
        print(f"{Colors.GREEN}Thank you for using the dashboard!{Colors.ENDC}")
        print()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    main()
