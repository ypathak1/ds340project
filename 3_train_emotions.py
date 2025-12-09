#!/usr/bin/env python3
"""
Emotion Model Training Script
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from pathlib import Path
import shutil

# Import the model saver utility
try:
    from model_saver import ModelSaver
    USE_MODEL_SAVER = True
except ImportError:
    print("model_saver.py not found!")
    USE_MODEL_SAVER = False

print("=" * 70)
print("CUSTOM EMOTION MODEL TRAINING")
print("=" * 70)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

LABELS = np.array(['happy', 'sad', 'neutral', 'angry'])
DATA_DIR = Path('.')

# Neural network parameters (from your parameter tuning)
ARCHITECTURE = [512, 256, 128]
DROPOUT_RATES = [0.4, 0.3, 0.2]
BATCH_SIZE = 16
EPOCHS = 50
VALIDATION_SPLIT = 0.2

print(f"\nModel Configuration:")
print(f"  Architecture: {ARCHITECTURE}")
print(f"  Dropout: {DROPOUT_RATES}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")

# ==============================================================================
# LOAD TRAINING DATA
# ==============================================================================

print("\n" + "-" * 70)
print("Loading Training Data")
print("-" * 70)

X_data = []
y_data = []

for idx, emotion in enumerate(LABELS):
    filename = f'emotion_{emotion}.npy'
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: {filename} not found - skipping {emotion}")
        continue
    
    try:
        samples = np.load(str(filepath))
        print(f"Loaded {len(samples):3d} samples for '{emotion}'")
        
        X_data.append(samples)
        y_data.extend([idx] * len(samples))
        
    except Exception as e:
        print(f" Error loading {filename}: {e}")

if not X_data:
    print("\nERROR: No training data found!")
    print("\nPlease collect training data first:")
    print("  python 1_collect_emotions.py")
    exit(1)

# Combine all data
X = np.vstack(X_data)
y = np.array(y_data)

print(f"\n📊 Total Dataset:")
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(LABELS)}")

# Check for class imbalance
print(f"\nClass Distribution:")
for idx, emotion in enumerate(LABELS):
    count = np.sum(y == idx)
    percentage = (count / len(y)) * 100
    print(f"  {emotion:8s}: {count:3d} samples ({percentage:5.1f}%)")

# Compute class weights to handle imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}

print(f"\n Class Weights (for balancing):")
for idx, emotion in enumerate(LABELS):
    print(f"  {emotion:8s}: {class_weights[idx]:.3f}")

# ==============================================================================
# TRAIN-TEST SPLIT
# ==============================================================================

print("\n" + "-" * 70)
print("Creating Train/Validation Split")
print("-" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=VALIDATION_SPLIT, 
    random_state=42,
    stratify=y  # Maintain class distribution in split
)

print(f"Training set:   {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# ==============================================================================
# BUILD MODEL
# ==============================================================================

print("\n" + "-" * 70)
print("Building Neural Network")
print("-" * 70)

model = keras.Sequential()

# Input layer
model.add(layers.Dense(ARCHITECTURE[0], activation='relu', input_shape=(X.shape[1],)))
model.add(layers.Dropout(DROPOUT_RATES[0]))

# Hidden layers
for units, dropout in zip(ARCHITECTURE[1:], DROPOUT_RATES[1:]):
    model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dropout(dropout))

# Output layer
model.add(layers.Dense(len(LABELS), activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# ==============================================================================
# TRAIN MODEL
# ==============================================================================

print("\n" + "-" * 70)
print("Training Model")
print("-" * 70)

# To prevent overfitting, use early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)

# ==============================================================================
# EVALUATE MODEL
# ==============================================================================

print("\n" + "-" * 70)
print("Evaluation Results")
print("-" * 70)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print(f"\nTraining Accuracy:   {train_acc:.2%}")
print(f"Validation Accuracy: {val_acc:.2%}")
print(f"Training Loss:       {train_loss:.4f}")
print(f"Validation Loss:     {val_loss:.4f}")

if train_acc - val_acc > 0.15:
    print("\n Warning: Overfitting potentially detected")
    print("   (Training accuracy significantly higher than validation)")

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

if USE_MODEL_SAVER:
    # Use the fancy model saver with backups and metadata
    saver = ModelSaver(model_dir="models")
    
    # Create informative notes
    notes = f"Trained on {len(X)} samples with {len(LABELS)} emotions. "
    notes += f"Final validation accuracy: {val_acc:.2%}. "
    notes += f"Architecture: {ARCHITECTURE}. "
    
    # Check for class imbalance in notes
    min_samples = min([np.sum(y == i) for i in range(len(LABELS))])
    max_samples = max([np.sum(y == i) for i in range(len(LABELS))])
    if max_samples > min_samples * 1.5:
        notes += "Note: Class imbalance present in training data."
    
    saver.save_model(
        model=model,
        labels=LABELS,
        training_history=history,
        notes=notes
    )
    
    print("\n Model saved with model_saver.py")
    
    # Maintain compatibility with existing pipeline (expects root-level files)
    try:
        shutil.copy2("models/custom_emotion_model.h5", "custom_emotion_model.h5")
        shutil.copy2("models/custom_emotion_labels.npy", "custom_emotion_labels.npy")
        print("\n📦 Copied model files to project root for calculator compatibility.")
    except Exception as copy_err:
        print(f"\n Warning: Could not copy model files to root directory: {copy_err}")

else:
    # Fallback to basic saving
    model.save('custom_emotion_model.h5')
    np.save('custom_emotion_labels.npy', LABELS)
    
    print("\n Model saved (basic mode)")
    print("   custom_emotion_model.h5")
    print("   custom_emotion_labels.npy")

# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if val_acc < 0.7:
    print("\n Validation accuracy is below 70%")
    print("\nSuggestions to improve:")
    print("  1. Collect more training samples (aim for 100+ per emotion)")
    print("  2. Ensure balanced data across all emotions")
    print("  3. Make sure facial expressions are clear and varied")
    print("  4. Use good lighting when collecting samples")

elif val_acc < 0.85:
    print("\n✓ Model performance is acceptable")
    print("\nTo further improve:")
    print("  1. Add more diverse facial expressions")
    print("  2. Collect samples under different lighting conditions")
    print("  3. Vary facial angles slightly")

else:
    print("\n Excellent model performance!")
    print(f"\nYour model achieved {val_acc:.2%} validation accuracy")
    print("This should work well in the real-time calculator!")


# Class balance check
for idx, emotion in enumerate(LABELS):
    count = np.sum(y == idx)
    if count < 30:
        print(f"\n  Low sample count for '{emotion}': only {count} samples")
        print(f"   Collect at least 20 more samples for better performance...")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("  1. Test the model in the calculator! Thanks for training :)")
