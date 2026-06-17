"""
Test if ImageDataGenerator can see your images.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = 'dataset'
CLASS_NAMES = ['clear_skin', 'mild', 'moderate', 'severe']

print("Testing TRAIN generator...")
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=CLASS_NAMES
)

print(f"\nFound {train_gen.samples} training images")
print(f"Class indices: {train_gen.class_indices}")

print("\nTesting VALIDATION generator...")
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATASET_PATH, 'validation'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=CLASS_NAMES
)

print(f"\nFound {val_gen.samples} validation images")

if train_gen.samples > 0 and val_gen.samples > 0:
    print("\n✅ SUCCESS! Generators are working correctly.")
    print("You can now run: python training/train.py")
else:
    print("\n❌ Still showing 0 images. Check folder paths.")