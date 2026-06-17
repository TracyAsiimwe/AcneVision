"""
Step 3: Data Augmentation
==========================
Apply rotation, flipping, zoom, brightness adjustment.
Handle class imbalance through oversampling.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 30,        # Random rotation ±30 degrees
    'horizontal_flip': True,      # Random horizontal flip
    'vertical_flip': False,       # No vertical flip (faces are upright)
    'zoom_range': [0.8, 1.2],     # Random zoom 80% to 120%
    'brightness_range': [0.7, 1.3],  # Brightness factor
    'shear_range': 10,            # Shear angle in degrees
    'fill_mode': 'nearest'        # Fill strategy for empty pixels
}

TARGET_COUNT_PER_CLASS = 1000  # Target images per class for balance

def apply_rotation(image, angle):
    """Rotate image by given angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), 
                             borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_flip(image, horizontal=True):
    """Flip image horizontally or vertically."""
    if horizontal:
        return cv2.flip(image, 1)  # Horizontal flip
    else:
        return cv2.flip(image, 0)   # Vertical flip

def apply_zoom(image, zoom_factor):
    """Zoom in/out while maintaining size."""
    h, w = image.shape[:2]
    
    # Calculate crop dimensions
    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)
    
    # Ensure minimum size
    new_h = max(new_h, 1)
    new_w = max(new_w, 1)
    
    # Random crop position
    y = random.randint(0, h - new_h) if new_h < h else 0
    x = random.randint(0, w - new_w) if new_w < w else 0
    
    # Crop and resize back
    cropped = image[y:y + new_h, x:x + new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed

def apply_brightness(image, factor):
    """Adjust image brightness."""
    # Convert to HSV for better brightness control
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    brightened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return brightened

def apply_shear(image, shear_angle):
    """Apply shear transformation."""
    h, w = image.shape[:2]
    shear_rad = np.deg2rad(shear_angle)
    
    matrix = np.array([
        [1, -np.tan(shear_rad), 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # Adjust for translation
    new_w = int(w + h * abs(np.tan(shear_rad)))
    matrix[0, 2] = (new_w - w) / 2
    
    sheared = cv2.warpAffine(image, matrix, (new_w, h), 
                            borderMode=cv2.BORDER_REFLECT)
    # Resize back to original
    sheared = cv2.resize(sheared, (w, h))
    return sheared

def random_augmentation(image):
    """
    Apply random combination of augmentations.
    Each augmentation has 50% chance of being applied.
    """
    augmented = image.copy()
    
    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-AUGMENTATION_CONFIG['rotation_range'],
                              AUGMENTATION_CONFIG['rotation_range'])
        augmented = apply_rotation(augmented, angle)
    
    # Random horizontal flip
    if AUGMENTATION_CONFIG['horizontal_flip'] and random.random() < 0.5:
        augmented = apply_flip(augmented, horizontal=True)
    
    # Random zoom
    if random.random() < 0.5:
        zoom = random.uniform(AUGMENTATION_CONFIG['zoom_range'][0],
                             AUGMENTATION_CONFIG['zoom_range'][1])
        augmented = apply_zoom(augmented, zoom)
    
    # Random brightness
    if random.random() < 0.5:
        brightness = random.uniform(AUGMENTATION_CONFIG['brightness_range'][0],
                                   AUGMENTATION_CONFIG['brightness_range'][1])
        augmented = apply_brightness(augmented, brightness)
    
    # Random shear
    if random.random() < 0.5:
        shear = random.uniform(-AUGMENTATION_CONFIG['shear_range'],
                              AUGMENTATION_CONFIG['shear_range'])
        augmented = apply_shear(augmented, shear)
    
    return augmented

def augment_class_images(class_dir, target_count):
    """
    Augment images in a class directory until reaching target count.
    """
    image_files = [f for f in os.listdir(class_dir) 
                  if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    
    current_count = len(image_files)
    
    if current_count >= target_count:
        print(f"[INFO] Class {os.path.basename(class_dir)} already has {current_count} images")
        return current_count
    
    print(f"[AUGMENT] Class {os.path.basename(class_dir)}: {current_count} -> {target_count}")
    
    images_needed = target_count - current_count
    generated = 0
    
    while generated < images_needed:
        # Pick random source image
        src_file = random.choice(image_files)
        src_path = os.path.join(class_dir, src_file)
        
        # Read image
        img = cv2.imread(src_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        aug_img = random_augmentation(img)
        
        # Save augmented image
        aug_filename = f"aug_{generated}_{src_file}"
        aug_path = os.path.join(class_dir, aug_filename)
        cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        
        generated += 1
    
    return target_count

def balance_dataset(dataset_path, target_count=TARGET_COUNT_PER_CLASS):
    """
    Balance all classes in train split by augmenting minority classes.
    """
    train_path = os.path.join(dataset_path, 'train')
    
    if not os.path.exists(train_path):
        print(f"[ERROR] Train path not found: {train_path}")
        return
    
    print("=" * 60)
    print("DATASET BALANCING VIA AUGMENTATION")
    print("=" * 60)
    
    for class_name in sorted(os.listdir(train_path)):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        augment_class_images(class_dir, target_count)
    
    print("[SUCCESS] Dataset balancing complete!")

def generate_augmented_samples(source_image_path, num_samples=5, output_dir=None):
    """
    Generate multiple augmented versions of a single image.
    Useful for testing augmentation pipeline.
    """
    img = cv2.imread(source_image_path)
    if img is None:
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    samples = [img]  # Original
    for i in range(num_samples):
        aug = random_augmentation(img)
        samples.append(aug)
        
        if output_dir:
            save_path = os.path.join(output_dir, f"aug_{i}.png")
            cv2.imwrite(save_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
    
    return samples

if __name__ == '__main__':
    # Balance the training dataset
    DATASET_PATH = '../dataset'
    balance_dataset(DATASET_PATH, target_count=1000)
    
    print("\n[AUGMENTATION COMPLETE]")