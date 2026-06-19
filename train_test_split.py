import os
import shutil
import random
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

dataset_path = "/Users/takneekmacmini/Documents/Ball Labelling Pipeline/Manual-Ball-Labelling-Pipeline/dataset"

images_dir = os.path.join(dataset_path, "images")
labels_dir = os.path.join(dataset_path, "labels")

output_dir = os.path.join(dataset_path, "Split_Ball_Dataset")

# =====================================================
# SPLIT RATIOS
# =====================================================

train_ratio = 0.70
val_ratio = 0.20
test_ratio = 0.10

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(output_dir, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(output_dir, split, "labels"),
        exist_ok=True
    )

# =====================================================
# COLLECT VALID IMAGE-LABEL PAIRS
# =====================================================

image_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]

pairs = []

for image_file in os.listdir(images_dir):

    if Path(image_file).suffix not in image_extensions:
        continue

    stem = Path(image_file).stem

    label_file = stem + ".txt"

    label_path = os.path.join(labels_dir, label_file)

    if os.path.exists(label_path):

        pairs.append((image_file, label_file))

print(f"Found {len(pairs)} valid image-label pairs")

# =====================================================
# SHUFFLE
# =====================================================

random.seed(42)
random.shuffle(pairs)

# =====================================================
# SPLIT
# =====================================================

n = len(pairs)

train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

train_pairs = pairs[:train_end]
val_pairs = pairs[train_end:val_end]
test_pairs = pairs[val_end:]

print(f"Train: {len(train_pairs)}")
print(f"Val  : {len(val_pairs)}")
print(f"Test : {len(test_pairs)}")

# =====================================================
# COPY FILES
# =====================================================

def copy_split(data, split_name):

    for image_file, label_file in data:

        shutil.copy2(
            os.path.join(images_dir, image_file),
            os.path.join(
                output_dir,
                split_name,
                "images",
                image_file
            )
        )

        shutil.copy2(
            os.path.join(labels_dir, label_file),
            os.path.join(
                output_dir,
                split_name,
                "labels",
                label_file
            )
        )

copy_split(train_pairs, "train")
copy_split(val_pairs, "val")
copy_split(test_pairs, "test")

print("\nDataset split completed successfully!")