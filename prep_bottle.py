import os
import shutil
import random

# Path to your extracted 'bottle' folder in Downloads
DOWNLOADS_BOTTLE = os.path.join(os.path.expanduser("~"), "Downloads", "bottle")
DEST_DIR = r"C:\yolo_project\bottle_cls_dataset"

CLASSES = ["broken_large", "broken_small", "contamination", "good"]

for cls in CLASSES:
    os.makedirs(os.path.join(DEST_DIR, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "val", cls), exist_ok=True)

    # Gather images from test/<cls> (and train/good for the 'good' class)
    images = []
    test_cls_dir = os.path.join(DOWNLOADS_BOTTLE, "test", cls)
    if os.path.exists(test_cls_dir):
        for f in os.listdir(test_cls_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(test_cls_dir, f))

    if cls == "good":
        train_good_dir = os.path.join(DOWNLOADS_BOTTLE, "train", "good")
        if os.path.exists(train_good_dir):
            for f in os.listdir(train_good_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(train_good_dir, f))

    # Shuffle and split 80% train, 20% val
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)

    for i, img_path in enumerate(images):
        split = "train" if i < split_idx else "val"
        # Prefix with folder name to avoid duplicate filenames (e.g., 000.png)
        parent_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        new_name = f"{parent_name}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(DEST_DIR, split, cls, new_name))

print("Dataset prepared successfully at:", DEST_DIR)