"""
Step 2: Image Preprocessing
============================
Resize images, normalize pixel values, prepare for CNN input.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
TARGET_SIZE = (224, 224)  # MobileNetV2 input size
NORMALIZATION_MODE = 'imagenet'  # 'imagenet', 'standard', or 'none'

def resize_image(image_path, target_size=TARGET_SIZE):
    """
    Resize image to target dimensions while maintaining aspect ratio.
    Uses padding to prevent distortion.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_w / w, target_h / h)
    
    # New dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create blank canvas and center the image
    result = np.full((target_h, target_w, 3), 128, dtype=np.uint8)  # Gray padding
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def normalize_image(image, mode='imagenet'):
    """
    Normalize pixel values.
    
    Modes:
    - 'imagenet': Subtract ImageNet mean [123.68, 116.78, 103.94] and divide by 255
    - 'standard': Scale to [0, 1]
    - 'none': No normalization (0-255)
    """
    image = image.astype(np.float32)
    
    if mode == 'imagenet':
        # ImageNet normalization for MobileNetV2
        mean = np.array([123.68, 116.78, 103.94])
        image = image - mean
        image = image / 255.0
    elif mode == 'standard':
        image = image / 255.0
    elif mode == 'none':
        pass  # Keep as uint8
    
    return image

def preprocess_single_image(image_path, target_size=TARGET_SIZE, normalize=True):
    """
    Complete preprocessing pipeline for a single image.
    Returns preprocessed numpy array ready for model input.
    """
    # Step 1: Resize
    img = resize_image(image_path, target_size)
    if img is None:
        return None
    
    # Step 2: Normalize
    if normalize:
        img = normalize_image(img, mode=NORMALIZATION_MODE)
    
    return img

def preprocess_dataset(source_dir, output_dir, target_size=TARGET_SIZE):
    """
    Batch preprocess entire dataset directory.
    Maintains folder structure in output.
    """
    print(f"[INFO] Preprocessing dataset from {source_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Target size: {target_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_processed = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path to maintain structure
        rel_path = os.path.relpath(root, source_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Process each image
        image_files = [f for f in files 
                      if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        for filename in tqdm(image_files, desc=f"Processing {rel_path}"):
            src_path = os.path.join(root, filename)
            dst_path = os.path.join(output_path, filename)
            
            # Preprocess and save
            processed = preprocess_single_image(src_path, target_size)
            if processed is not None:
                # Convert back to uint8 for saving if normalized
                if processed.dtype == np.float32:
                    save_img = (processed * 255).clip(0, 255).astype(np.uint8)
                else:
                    save_img = processed
                
                # Save as PNG to avoid compression artifacts
                cv2.imwrite(dst_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
                total_processed += 1
    
    print(f"\n[SUCCESS] Preprocessed {total_processed} images")
    return total_processed

def create_preprocessed_splits(base_dataset_path, output_base_path):
    """
    Preprocess train, validation, and test splits.
    """
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        source = os.path.join(base_dataset_path, split)
        output = os.path.join(output_base_path, split)
        
        if os.path.exists(source):
            preprocess_dataset(source, output)
        else:
            print(f"[WARNING] Split not found: {source}")

if __name__ == '__main__':
    # Example usage
    DATASET_PATH = '../dataset'
    PREPROCESSED_PATH = '../dataset_preprocessed'
    
    create_preprocessed_splits(DATASET_PATH, PREPROCESSED_PATH)
    print("[DONE] All preprocessing complete!")