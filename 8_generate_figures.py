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
    
    if not content:
        metrics_list = []
    else:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                metrics_list.extend(data)
            else:
                metrics_list.append(data)
        except json.JSONDecodeError:
            for line in content.split('\n'):
                if line.strip():
                    try:
                        metrics_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

if not metrics_list:
    print("Error: No valid data found in log file")
    print("Make sure you've played the game (press G in calculator)")
    exit(1)

print(f"[INFO] Loaded {len(metrics_list)} data snapshots")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Get the latest metrics for summary
latest = metrics_list[-1]

print("\n[INFO] Current game stats:")
print(f"  Games played so far: {latest.get('games_played', 0)}")
print(f"  Total lifetime score: {latest.get('total_score', 0)}")
print(f"  Current mode: {latest.get('mode', 'CALCULATOR')}")
print(f"  Logged samples: {len(metrics_list)}")

# Prepare aggregated histories
sample_indices = list(range(len(metrics_list)))

vote_series = {}
agreement_series = []
score_series = []
games_played_series = []
fps_series = []
frame_time_series = []
emotion_series = []
game_only_scores = []
game_rounds = []

for entry in metrics_list:
    rv = entry.get('recent_votes')
    if isinstance(rv, dict):
        for model, val in rv.items():
            vote_series.setdefault(model, []).append(val)
    agreement_series.append(entry.get('vote_agreement', 0))
    score_series.append(entry.get('total_score', 0))
    games_played_series.append(entry.get('games_played', 0))
    fps_series.append(entry.get('fps', 0))
    frame_time_series.append(entry.get('frame_time', 0))
    emotion_series.append(entry.get('current_emotion', 'neutral'))
    
    if entry.get('mode') == 'GAME MODE':
        game_only_scores.append(entry.get('game_score', 0))
        game_rounds.append(entry.get('round', len(game_rounds)))

# Figure 1: Average model vote agreement across entire log
if vote_series:
    plt.figure(figsize=(10, 6))
    models = list(vote_series.keys())
    avg_agreements = [np.mean(vote_series[m]) for m in models]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = plt.bar(models, avg_agreements, color=colors[:len(models)])
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Agreement (%)', fontsize=12)
    plt.title('Average Model Voting Agreement Across Sessions', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_model_agreement.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure1_model_agreement.png")
    plt.close()

# Figure 2: Vote agreement trend
if len(agreement_series) > 1:
    plt.figure(figsize=(12, 5))
    plt.plot(sample_indices, agreement_series, color='#45B7D1', linewidth=2)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Agreement (%)', fontsize=12)
    plt.title('Ensemble Vote Agreement Over Time', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_ensemble_agreement.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure2_ensemble_agreement.png")
    plt.close()

# Figure 3: Total score & games played evolution
if len(score_series) > 1:
    plt.figure(figsize=(12, 6))
    plt.plot(sample_indices, score_series, label='Total Score', color='#FF6B6B', linewidth=2)
    plt.plot(sample_indices, games_played_series, label='Games Played', color='#4ECDC4', linewidth=2)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Game Mode Progression Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_game_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure3_game_summary.png")
    plt.close()

# Figure 4: System performance history (FPS and frame time)
if len(fps_series) > 1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(sample_indices, fps_series, color='#96CEB4')
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.set_title('FPS Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sample_indices, frame_time_series, color='#FF6B6B')
    ax2.set_ylabel('Frame Time (ms)', fontsize=12)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_title('Frame Processing Time Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_system_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: figure4_system_performance.png")
    plt.close()

# Figure 5: Emotion distribution
emotion_counts = Counter(emotion_series)
if emotion_counts:
    plt.figure(figsize=(8, 6))
    emotions = list(emotion_counts.keys())
    counts = [emotion_counts[e] for e in emotions]
    colors = ['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B6B']
    
    plt.bar(emotions, counts, color=colors[:len(emotions)], edgecolor='black')
    plt.ylabel('Occurrences', fontsize=12)
    plt.title('Detected Emotions Across Sessions', fontsize=14, fontweight='bold')
    for i, val in enumerate(counts):
        plt.text(i, val + max(1, val * 0.01), str(val), ha='center', fontsize=11, fontweight='bold')
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
print("Figure 1: Average model voting agreement across the entire log")
print("Figure 2: Ensemble vote agreement trend over recorded samples")
print("Figure 3: Total score and games played progression")
print("Figure 4: FPS and frame time history while running the calculator")
print("Figure 5: Distribution of detected emotions across all sessions")
print()

print("NOTE: For more detailed game figures (score progression, solve times, etc.),")
print("play game mode (press G) for several rounds to generate more data.")
print()
