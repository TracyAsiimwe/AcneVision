"""
AcneVision — Improved Training Script v2
Fixes: class imbalance, over-prediction of dominant classes,
       poor Moderate/Clear detection.
Location: AcneVision/training/train.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ========================= PATHS =========================
THIS_FILE    = os.path.abspath(__file__)
TRAINING_DIR = os.path.dirname(THIS_FILE)
BASE_DIR     = os.path.dirname(TRAINING_DIR)

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
IMAGES_DIR  = os.path.join(BASE_DIR, 'static', 'images')

MODEL_KERAS = os.path.join(MODEL_DIR, 'acne_model.keras')
MODEL_H5    = os.path.join(MODEL_DIR, 'acne_model.h5')

# ========================= HYPERPARAMETERS =========================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
LR         = 1e-4

# Phase 1: train top layers only (base frozen)
EPOCHS_P1  = 20
# Phase 2: fine-tune last N base layers
EPOCHS_P2  = 60
FINETUNE_LAYERS = 40   # unfreeze last 40 layers of MobileNetV2


# ================================================================
# STEP 0: DATASET DIAGNOSIS
# ================================================================
def diagnose():
    print("\n" + "=" * 70)
    print("DATASET DIAGNOSTICS")
    print("=" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"FATAL: dataset/ folder missing at: {DATASET_DIR}")
        sys.exit(1)

    class_counts = {}
    for split in ['train', 'validation']:
        sp = os.path.join(DATASET_DIR, split)
        print(f"\n--- {split.upper()} ---")
        if not os.path.exists(sp):
            print(f"  MISSING: {sp}")
            continue

        folders = sorted([d for d in os.listdir(sp)
                          if os.path.isdir(os.path.join(sp, d)) and not d.startswith('.')])
        total = 0
        split_counts = {}
        for cls in folders:
            cp   = os.path.join(sp, cls)
            imgs = [f for f in os.listdir(cp)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            print(f"  {cls:20}: {len(imgs):4} images")
            split_counts[cls] = len(imgs)
            total += len(imgs)
        print(f"  {'TOTAL':20}: {total:4} images")
        class_counts[split] = split_counts

    # Balance warning
    if 'train' in class_counts:
        counts = list(class_counts['train'].values())
        if counts:
            ratio = max(counts) / (min(counts) + 1)
            print(f"\n[BALANCE CHECK] Max/Min ratio = {ratio:.1f}x")
            if ratio > 2.0:
                print("⚠️  WARNING: Dataset is imbalanced (ratio > 2x).")
                print("   Weighted loss + augmentation will compensate,")
                print("   but consider adding images to under-represented classes.")
            else:
                print("✅  Dataset balance looks good.")
    print()
    return class_counts


# ================================================================
# STEP 1: DATA GENERATORS
# ================================================================
def create_generators():
    train_path = os.path.join(DATASET_DIR, 'train')
    val_path   = os.path.join(DATASET_DIR, 'validation')

    # Aggressive augmentation to combat overfitting on small datasets
    train_aug = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        channel_shift_range=15.0,
        fill_mode='nearest',
    )

    val_aug = ImageDataGenerator(rescale=1. / 255)

    print("[INFO] Creating training generator...")
    train_gen = train_aug.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )

    print("[INFO] Creating validation generator...")
    val_gen = val_aug.flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    print(f"\nTrain samples     : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes           : {train_gen.class_indices}\n")

    return train_gen, val_gen


# ================================================================
# STEP 2: FOCAL LOSS
# Focal loss down-weights easy examples so the model doesn't
# just learn the dominant class. Key fix for imbalanced datasets.
# ================================================================
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss: reduces the relative loss for well-classified examples
    and focuses training on hard, misclassified examples.
    gamma=2 is standard; increase to 3–4 if imbalance is severe.
    """
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce      = -y_true * tf.math.log(y_pred)
        weight  = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        fl      = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return loss_fn


# ================================================================
# STEP 3: MODEL ARCHITECTURE
# ================================================================
def build_model(num_classes):
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False   # freeze for Phase 1

    inputs = keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head — deeper with stronger regularisation
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs, outputs), base


# ================================================================
# STEP 4: CALLBACKS
# ================================================================
def get_callbacks(phase=1):
    os.makedirs(MODEL_DIR, exist_ok=True)
    return [
        ModelCheckpoint(
            MODEL_KERAS,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            # Phase 1: less patience so we move to fine-tuning quickly
            patience=8 if phase == 1 else 14,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=4,
            min_lr=1e-9,
            verbose=1,
        ),
    ]


# ================================================================
# STEP 5: EVALUATION & CONFUSION MATRIX
# ================================================================
def evaluate_model(model, val_gen, class_indices):
    print("\n" + "=" * 60)
    print("PER-CLASS EVALUATION")
    print("=" * 60)

    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes

    # Map index → class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names  = [idx_to_class[i] for i in range(len(class_indices))]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — Validation Set', fontsize=14, pad=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(IMAGES_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[INFO] Confusion matrix saved to static/images/confusion_matrix.png")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        mask    = y_true == i
        correct = np.sum(y_pred[mask] == i)
        total   = np.sum(mask)
        pct     = 100 * correct / total if total > 0 else 0
        bar     = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
        print(f"  {name:20} [{bar}] {pct:5.1f}%  ({correct}/{total})")

    return y_pred, y_true


# ================================================================
# STEP 6: TRAINING HISTORY PLOT
# ================================================================
def plot_history(h1, h2):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    acc     = h1.history['accuracy']     + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss    = h1.history['loss']         + h2.history['loss']
    val_loss= h1.history['val_loss']     + h2.history['val_loss']

    ep_split = len(h1.history['accuracy'])   # where Phase 2 starts

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('AcneVision — Training History', fontsize=14)

    for ax, train_vals, val_vals, title in zip(
        axes,
        [acc, loss],
        [val_acc, val_loss],
        ['Accuracy', 'Loss'],
    ):
        ax.plot(train_vals, label='Train', linewidth=2)
        ax.plot(val_vals,   label='Validation', linewidth=2)
        ax.axvline(ep_split, color='red', linestyle='--',
                   alpha=0.6, label='Fine-tune starts')
        ax.set_title(f'Model {title}')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'training_history.png'), dpi=200)
    plt.close()
    print("[INFO] Training history saved to static/images/training_history.png")


# ================================================================
# MAIN TRAINING LOOP
# ================================================================
def train():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    diagnose()
    train_gen, val_gen = create_generators()
    num_classes = len(train_gen.class_indices)

    # ── Compute class weights ──────────────────────────────
    # This makes the loss penalise mistakes on rare classes more.
    cw_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight_dict = dict(enumerate(cw_array))
    print(f"[INFO] Class weights: { {k: round(v,3) for k,v in class_weight_dict.items()} }")

    model, base = build_model(num_classes)
    model.summary()

    # ══════════════════════════════════════════════════════
    # PHASE 1 — Train head only (base frozen)
    # Uses focal loss + class weights to fight imbalance.
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Training Classification Head (Base Frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy'],
    )

    h1 = model.fit(
        train_gen,
        epochs=EPOCHS_P1,
        validation_data=val_gen,
        callbacks=get_callbacks(phase=1),
        class_weight=class_weight_dict,
    )

    best_p1_acc = max(h1.history['val_accuracy'])
    print(f"\n[Phase 1] Best val accuracy: {best_p1_acc:.4f}")

    # ══════════════════════════════════════════════════════
    # PHASE 2 — Unfreeze last N layers for fine-tuning
    # Lower LR to avoid destroying pre-trained features.
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 2: Fine-tuning Last {FINETUNE_LAYERS} Layers")
    print("=" * 60)

    base.trainable = True
    for layer in base.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"[INFO] Trainable layers: {trainable_count}")

    model.compile(
        optimizer=keras.optimizers.Adam(LR / 10),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy'],
    )

    h2 = model.fit(
        train_gen,
        epochs=EPOCHS_P1 + EPOCHS_P2,
        initial_epoch=len(h1.history['accuracy']),
        validation_data=val_gen,
        callbacks=get_callbacks(phase=2),
        class_weight=class_weight_dict,
    )

    # ══════════════════════════════════════════════════════
    # SAVE MODEL
    # ══════════════════════════════════════════════════════
    model.save(MODEL_KERAS)
    print(f"\n[INFO] Model saved: {MODEL_KERAS}")

    try:
        model.save(MODEL_H5)
        print(f"[INFO] Model saved: {MODEL_H5}")
    except Exception as e:
        print(f"[WARNING] .h5 save failed: {e}")

    # ══════════════════════════════════════════════════════
    # EVALUATE — per-class breakdown + confusion matrix
    # ══════════════════════════════════════════════════════
    evaluate_model(model, val_gen, train_gen.class_indices)
    plot_history(h1, h2)

    # ══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════
    best_overall = max(h2.history['val_accuracy'])
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_overall:.4f}  ({best_overall*100:.1f}%)")
    print("\nClass order in model output (use in prediction.py):")
    ordered = sorted(train_gen.class_indices.items(), key=lambda x: x[1])
    print("CLASS_NAMES =", [name for name, _ in ordered])
    print("\nNext steps:")
    print("  1. Check confusion_matrix.png — every class should have some correct predictions.")
    print("  2. If any class is at 0% accuracy, add more images for that class and retrain.")
    print("  3. If val accuracy < 60%, consider collecting more data or increasing epochs.")
    print("=" * 70)


# ================================================================
if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU detected: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("[INFO] No GPU detected — running on CPU (will be slower)")

    train()