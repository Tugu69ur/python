import os
import shutil

# Paths
src_root = "synth_dataset"      # original folder with subfolders per class
dst_root = "yolo_dataset"       # combined YOLO dataset
os.makedirs(os.path.join(dst_root, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "labels"), exist_ok=True)

# Counter for renaming images to avoid conflicts
count = 0

for class_folder in os.listdir(src_root):
    class_path = os.path.join(src_root, class_folder)
    if not os.path.isdir(class_path):
        continue
    images_path = os.path.join(class_path, "images")
    labels_path = os.path.join(class_path, "labels")
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        continue

    for img_file in os.listdir(images_path):
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        label_file = img_file.rsplit(".", 1)[0] + ".txt"

        # New names
        new_img_name = f"{count:05d}.png"
        new_label_name = f"{count:05d}.txt"

        # Copy image
        shutil.copy(os.path.join(images_path, img_file),
                    os.path.join(dst_root, "images", new_img_name))
        # Copy label
        shutil.copy(os.path.join(labels_path, label_file),
                    os.path.join(dst_root, "labels", new_label_name))

        count += 1

print(f"✅ Combined {count} images and labels into {dst_root}")
