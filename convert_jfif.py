from PIL import Image
import os

folder = r"dataset\train\severe"

for file in os.listdir(folder):
    if file.lower().endswith(".jfif"):
        old_path = os.path.join(folder, file)

        img = Image.open(old_path)

        new_name = os.path.splitext(file)[0] + ".jpg"
        new_path = os.path.join(folder, new_name)

        img.convert("RGB").save(new_path, "JPEG")

        print(f"Converted: {file} -> {new_name}")