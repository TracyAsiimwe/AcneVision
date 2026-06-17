"""
Dataset Diagnostic Tool
Run this to see exactly what the training code sees.
"""

import os

DATASET_PATH = 'dataset'
EXPECTED_CLASSES = ['clear_skin', 'mild', 'moderate', 'severe']
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')

print("=" * 60)
print("DATASET DIAGNOSTIC REPORT")
print("=" * 60)

for split in ['train', 'validation', 'test']:
    split_path = os.path.join(DATASET_PATH, split)
    print(f"\n📁 {split.upper()}/")
    
    if not os.path.exists(split_path):
        print(f"   ❌ FOLDER DOES NOT EXIST: {split_path}")
        continue
    
    # List what's inside
    contents = os.listdir(split_path)
    print(f"   Contents: {contents}")
    
    for cls in EXPECTED_CLASSES:
        cls_path = os.path.join(split_path, cls)
        print(f"\n   📂 {cls}/")
        
        if not os.path.exists(cls_path):
            print(f"      ❌ FOLDER MISSING!")
            continue
        
        # Check all files
        all_files = os.listdir(cls_path)
        print(f"      Total items: {len(all_files)}")
        
        # Filter valid images
        images = [f for f in all_files if f.lower().endswith(VALID_EXTENSIONS)]
        print(f"      Valid images: {len(images)}")
        
        # Show first 5 files found (any type)
        if all_files:
            print(f"      First 5 items: {all_files[:5]}")
        
        # Warn about invalid files
        invalid = [f for f in all_files if not f.lower().endswith(VALID_EXTENSIONS)]
        if invalid:
            print(f"      ⚠️  Skipped (wrong format): {invalid[:3]}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

total_images = 0
for split in ['train', 'validation', 'test']:
    for cls in EXPECTED_CLASSES:
        cls_path = os.path.join(DATASET_PATH, split, cls)
        if os.path.exists(cls_path):
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(VALID_EXTENSIONS)]
            total_images += len(images)

print(f"Total valid images found: {total_images}")

if total_images == 0:
    print("\n🔴 PROBLEM: No valid images found!")
    print("   Possible fixes:")
    print("   1. Check folder names match: clear_skin, mild, moderate, severe")
    print("   2. Check image extensions are: .jpg, .jpeg, .png, .bmp")
    print("   3. Make sure images are directly inside class folders")
else:
    print(f"\n✅ Found {total_images} images. You can now run training!")