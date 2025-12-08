#!/usr/bin/env python3
"""
Model Saving and Loading Utilities for DS340 Project
Ensures models are properly saved for team collaboration via repo

This module provides robust saving/loading with:
- Automatic backup of previous models
- Timestamp tracking
- Validation after saving
- Error handling

Author: Jennifer Ji, Yana Pathak
Course: DS340
"""

import os
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
import json

class ModelSaver:
    """Handles saving and loading models with versioning and validation"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Backup directory for previous versions
        self.backup_dir = self.model_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.model_path = self.model_dir / "custom_emotion_model.h5"
        self.labels_path = self.model_dir / "custom_emotion_labels.npy"
        self.metadata_path = self.model_dir / "model_metadata.json"
    
    def save_model(self, model, labels, training_history=None, notes=""):
        """
        Save model with automatic backup and metadata
        
        Args:
            model: Keras model to save
            labels: numpy array of label names
            training_history: Optional training history dict
            notes: Optional notes about this model version
        """
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        # Create backup of existing model if it exists
        if self.model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_model = self.backup_dir / f"custom_emotion_model_{timestamp}.h5"
            backup_labels = self.backup_dir / f"custom_emotion_labels_{timestamp}.npy"
            
            print(f"📦 Creating backup of previous model:")
            print(f"   {backup_model.name}")
            
            shutil.copy2(self.model_path, backup_model)
            shutil.copy2(self.labels_path, backup_labels)
        
        # Save new model
        print(f"\n💾 Saving new model:")
        print(f"   Model: {self.model_path}")
        print(f"   Labels: {self.labels_path}")
        
        model.save(str(self.model_path))
        np.save(str(self.labels_path), labels)
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "labels": labels.tolist(),
            "notes": notes,
            "model_path": str(self.model_path),
            "labels_path": str(self.labels_path)
        }
        
        if training_history:
            metadata["training_history"] = {
                "final_accuracy": float(training_history.history['accuracy'][-1]),
                "final_loss": float(training_history.history['loss'][-1]),
                "epochs": len(training_history.history['accuracy'])
            }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata: {self.metadata_path}")
        
        # Validate the save
        print(f"\n✅ Validating saved files...")
        if self.validate_model_files():
            print("   All files saved successfully and validated!")
        else:
            print("   ⚠️ Warning: Validation failed - check files manually")
        
        print("\n" + "="*70)
        print("MODEL SAVED SUCCESSFULLY")
        print("="*70)
        print(f"\nModel location: {self.model_path.absolute()}")
        print("This model is now ready to be committed to your repo!")
        print()
    
    def validate_model_files(self):
        """Validate that all necessary files exist and are readable"""
        try:
            # Check model file exists and is not empty
            if not self.model_path.exists():
                print(f"   ❌ Model file not found: {self.model_path}")
                return False
            
            if self.model_path.stat().st_size == 0:
                print(f"   ❌ Model file is empty: {self.model_path}")
                return False
            
            # Check labels file
            if not self.labels_path.exists():
                print(f"   ❌ Labels file not found: {self.labels_path}")
                return False
            
            # Try to load labels
            labels = np.load(str(self.labels_path))
            if len(labels) == 0:
                print(f"   ❌ Labels array is empty")
                return False
            
            print(f"   ✓ Model file exists ({self.model_path.stat().st_size} bytes)")
            print(f"   ✓ Labels file exists ({len(labels)} labels)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return False
    
    def load_model(self):
        """
        Load the most recent model
        
        Returns:
            (model, labels) tuple or (None, None) if not found
        """
        from tensorflow import keras
        
        if not self.model_path.exists():
            print(f"No model found at {self.model_path}")
            return None, None
        
        try:
            model = keras.models.load_model(str(self.model_path))
            labels = np.load(str(self.labels_path))
            
            print(f"✓ Loaded model from {self.model_path}")
            print(f"✓ Loaded {len(labels)} labels")
            
            return model, labels
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    def get_model_info(self):
        """Get metadata about the current model"""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def list_backups(self):
        """List all backup models"""
        backups = sorted(self.backup_dir.glob("custom_emotion_model_*.h5"))
        
        if not backups:
            print("No backup models found")
            return []
        
        print(f"\nFound {len(backups)} backup models:")
        for i, backup in enumerate(backups, 1):
            timestamp_str = backup.stem.split('_', 3)[-1]
            print(f"  {i}. {backup.name} ({timestamp_str})")
        
        return backups


# ==============================================================================
# INTEGRATION WITH TRAINING SCRIPT
# ==============================================================================

def save_trained_model_example():
    """
    Example of how to integrate with your training script
    
    Add this to your 3_train_emotions.py file after training:
    """
    
    example_code = '''
# At the end of your 3_train_emotions.py file, replace:
#   model.save('custom_emotion_model.h5')
#   np.save('custom_emotion_labels.npy', LABELS)

# With this:
from model_saver import ModelSaver

# Create saver
saver = ModelSaver(model_dir="models")

# Save with metadata
saver.save_model(
    model=model,
    labels=LABELS,
    training_history=history,  # from model.fit()
    notes="Trained on balanced dataset with 100 samples per emotion"
)

# Now the model is saved with:
# - Automatic backup of previous version
# - Metadata (timestamp, accuracy, etc.)
# - Validation to ensure files are correct
# - Ready to commit to repo!
'''
    
    print(example_code)


if __name__ == "__main__":
    # Demonstrate usage
    print("Model Saver Utility for DS340 Project")
    print("=" * 70)
    print()
    print("This utility ensures your trained models are properly saved")
    print("for team collaboration through your git repository.")
    print()
    print("FEATURES:")
    print("  ✓ Automatic backup of previous models")
    print("  ✓ Timestamp tracking")
    print("  ✓ Metadata logging (accuracy, training info)")
    print("  ✓ Validation after saving")
    print("  ✓ Easy loading in other scripts")
    print()
    
    # Show example integration
    save_trained_model_example()
    
    # Show current model info if it exists
    saver = ModelSaver()
    info = saver.get_model_info()
    
    if info:
        print("\n" + "=" * 70)
        print("CURRENT MODEL INFORMATION:")
        print("=" * 70)
        print(f"Saved: {info.get('timestamp', 'Unknown')}")
        print(f"Labels: {info.get('labels', [])}")
        
        if 'training_history' in info:
            hist = info['training_history']
            print(f"Final Accuracy: {hist.get('final_accuracy', 0):.2%}")
            print(f"Training Epochs: {hist.get('epochs', 0)}")
        
        print()
    
    # List backups
    saver.list_backups()
