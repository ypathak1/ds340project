"""
STEP 3: Train Custom Emotion Model
Trains YOUR personalized emotion recognition model
"""

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("STEP 3: TRAINING CUSTOM EMOTION MODEL")
print("=" * 70)

is_init = False
size = -1
label = []
dictionary = {}
c = 0

print("\n[INFO] Loading emotion data files...")

for i in os.listdir():
    if i.startswith("emotion_") and i.endswith(".npy"):
        emotion_name = i.replace("emotion_", "").replace(".npy", "")
        print(f"  ✓ Found: {emotion_name}")
        data = np.load(i)
        current_size = data.shape[0]
        
        if not is_init:
            is_init = True
            X = data
            size = current_size
            y = np.array([emotion_name] * current_size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([emotion_name] * current_size).reshape(-1, 1)))
        
        label.append(emotion_name)
        dictionary[emotion_name] = c
        c += 1

if not is_init:
    print("\n[ERROR] No emotion data found!")
    print("Please run 1_collect_emotions.py first!")
    exit(1)

print(f"\n[INFO] Loaded {len(label)} emotions: {label}")
print(f"[INFO] Total samples: {X.shape[0]}")
print(f"[INFO] Features per sample: {X.shape[1]}")

# Convert labels
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

# Shuffle
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
X_shuffled = X[cnt]
y_shuffled = y[cnt]

print("\n[INFO] Building custom emotion model...")
print("[INFO] Architecture: Deep network with batch normalization")

# Enhanced architecture
ip = Input(shape=(X.shape[1],))
x = Dense(512, activation="relu")(ip)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
op = Dense(y.shape[1], activation="softmax")(x)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

print("\n[INFO] Training (50 epochs with early stopping)...\n")

history = model.fit(
    X_shuffled, y_shuffled,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model.save("custom_emotion_model.h5")
np.save("custom_emotion_labels.npy", np.array(label))

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "=" * 70)
print("[SUCCESS] Custom model trained!")
print("=" * 70)
print(f"[ACCURACY] Training: {final_train_acc*100:.2f}%")
print(f"[ACCURACY] Validation: {final_val_acc*100:.2f}%")
print(f"[SAVED] custom_emotion_model.h5")
print(f"[SAVED] custom_emotion_labels.npy")
print("\nNext: Run 4_run_calculator.py!\n")
