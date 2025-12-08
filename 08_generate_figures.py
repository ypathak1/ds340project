import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
import time

# Create output directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Load the realtime metrics log
log_file = Path("logs/realtime_metrics.json")

if not log_file.exists():
    print("Error: logs/realtime_metrics.json not found")
    print("Run 4_run_calculator_with_game.py and press G to play game mode first")
    exit(1)

# Parse the log file
print("[INFO] Reading game data...")

# The file might be a single JSON object or multiple lines
metrics_list = []
with open(log_file, 'r') as f:
    content = f.read().strip()
    
    # Try parsing as single JSON first
    try:
        data = json.loads(content)
        metrics_list.append(data)
    except:
        # Try parsing as JSONL (one JSON per line)
        for line in content.split('\n'):
            if line.strip():
                try:
                    metrics_list.append(json.loads(line))
                except:
                    continue

if not metrics_list:
    print("Error: No valid data found in log file")
    print("Make sure you've played the game (press G in calculator)")
    exit(1)

print(f"[INFO] Loaded {len(metrics_list)} data snapshots")

# Since we only have one snapshot, let's create DEMO figures
# showing what the system CAN track (even if we don't have full game data yet)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Get the latest metrics
latest = metrics_list[-1]

print("\n[INFO] Current game stats:")
print(f"  Games played: {latest.get('games_played', 0)}")
print(f"  Total score: {latest.get('total_score', 0)}")
print(f"  Current mode: {latest.get('mode', 'CALCULATOR')}")

# Figure 1: Model Vote Agreement (from current data)
vote_data = latest.get('recent_votes', {})
if vote_data and any(vote_data.values()):
    plt.figure(figsize=(10, 6))
    models = list(vote_data.keys())
    accuracies = list(vote_data.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = plt.bar(models, accuracies, color=colors[:len(models)])
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Agreement Rate (%)', fontsize=12)
    plt.title('Model Voting Agreement in Real-Time', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_model_agreement.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure1_model_agreement.png")
    plt.close()

# Figure 2: Overall Vote Agreement
agreement = latest.get('vote_agreement', 0)
plt.figure(figsize=(8, 6))
plt.bar(['Ensemble Vote Agreement'], [agreement], color='#45B7D1', width=0.5)
plt.ylim(0, 100)
plt.ylabel('Agreement Percentage (%)', fontsize=12)
plt.title('Ensemble Voting Stability', fontsize=14, fontweight='bold')
plt.text(0, agreement + 2, f'{agreement:.1f}%', ha='center', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'figure2_ensemble_agreement.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: figure2_ensemble_agreement.png")
plt.close()

# Figure 3: Game Performance Summary (if games were played)
games_played = latest.get('games_played', 0)
total_score = latest.get('total_score', 0)

if games_played > 0:
    plt.figure(figsize=(10, 6))
    
    metrics_names = ['Games Played', 'Total Score']
    metrics_values = [games_played, total_score]
    
    bars = plt.bar(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4'])
    plt.ylabel('Count', fontsize=12)
    plt.title('Game Mode Performance Summary', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_game_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure3_game_summary.png")
    plt.close()

# Figure 4: System Performance Metrics
fps = latest.get('fps', 0)
frame_time = latest.get('frame_time', 0)

plt.figure(figsize=(10, 6))

metrics_names = ['FPS', 'Frame Time (ms)']
metrics_values = [fps, frame_time]
colors = ['#96CEB4', '#FF6B6B']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# FPS
ax1.bar(['FPS'], [fps], color='#96CEB4', width=0.5)
ax1.set_ylabel('Frames Per Second', fontsize=12)
ax1.set_title('Real-Time Performance - FPS', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(30, fps * 1.2))
ax1.text(0, fps + 0.5, f'{fps:.1f}', ha='center', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Frame Time
ax2.bar(['Frame Time'], [frame_time], color='#FF6B6B', width=0.5)
ax2.set_ylabel('Milliseconds', fontsize=12)
ax2.set_title('Real-Time Performance - Frame Time', fontsize=12, fontweight='bold')
ax2.text(0, frame_time + 5, f'{frame_time:.1f}ms', ha='center', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'figure4_system_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: figure4_system_performance.png")
plt.close()

# Figure 5: Current Emotion Detection
current_emotion = latest.get('current_emotion', 'neutral')
emotion_colors = {
    'happy': '#FFD93D',
    'sad': '#6BCB77', 
    'neutral': '#4D96FF',
    'angry': '#FF6B6B'
}

plt.figure(figsize=(8, 6))
emotions = ['happy', 'sad', 'neutral', 'angry']
values = [100 if e == current_emotion else 0 for e in emotions]
colors_list = [emotion_colors[e] for e in emotions]

bars = plt.bar(emotions, values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
plt.ylim(0, 100)
plt.ylabel('Detection Status', fontsize=12)
plt.xlabel('Emotion', fontsize=12)
plt.title(f'Current Emotion Detection: {current_emotion.upper()}', fontsize=14, fontweight='bold')
plt.xticks(fontsize=11)

# Highlight detected emotion
for i, (bar, val) in enumerate(zip(bars, values)):
    if val > 0:
        bar.set_linewidth(3)
        plt.text(i, val + 5, '✓ DETECTED', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'figure5_current_emotion.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: figure5_current_emotion.png")
plt.close()

print(f"\n{'='*70}")
print("FIGURES GENERATED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"\nAll figures saved to: {output_dir}/\n")

print("Figure descriptions for your paper:")
print("-" * 70)
print("Figure 1: Model voting agreement rates showing individual model reliability")
print("Figure 2: Overall ensemble voting stability during real-time operation")
if games_played > 0:
    print("Figure 3: Game mode performance summary across all sessions")
print("Figure 4: System performance metrics (FPS and frame processing time)")
print("Figure 5: Real-time emotion detection showing current system state")
print()

print("NOTE: For more detailed game figures (score progression, solve times, etc.),")
print("play game mode (press G) for several rounds to generate more data.")
print()