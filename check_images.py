from PIL import Image
import os

folder = r"dataset\train\severe"

bad = []

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        bad.append(file)

print("\nBad files:")
for f in bad:
    print(f)

print(f"\nTotal bad files: {len(bad)}")