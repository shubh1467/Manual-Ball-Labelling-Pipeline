import cv2
import numpy as np
import os
import shutil

# ==================================
# PATHS
# ==================================

FRAMES_DIR = "extracted_frames"
MASKS_DIR = "masks"

OUTPUT_DIR = "dataset"

IMAGES_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

LABELS_DIR = os.path.join(
    OUTPUT_DIR,
    "labels"
)

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# ==================================
# FILES
# ==================================

mask_files = sorted([
    f for f in os.listdir(MASKS_DIR)
    if f.lower().endswith(".png")
])

print(
    f"Found {len(mask_files)} masks"
)

# ==================================
# LOOP
# ==================================

saved = 0

for mask_name in mask_files:

    mask_path = os.path.join(
        MASKS_DIR,
        mask_name
    )

    frame_name = (
        os.path.splitext(mask_name)[0]
        + ".png"
    )

    frame_path = os.path.join(
        FRAMES_DIR,
        frame_name
    )

    if not os.path.exists(frame_path):
        continue

    mask = cv2.imread(mask_path)

    if mask is None:
        continue

    h, w = mask.shape[:2]

    # ==================================
    # RED MASK DETECTION
    # ==================================

    hsv = cv2.cvtColor(
        mask,
        cv2.COLOR_BGR2HSV
    )

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    binary = mask1 + mask2

    # ==================================
    # CONTOUR
    # ==================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        continue

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if len(contour) < 5:
        continue

    # ==================================
    # FIT CIRCLE
    # ==================================

    (cx, cy), radius = cv2.minEnclosingCircle(
        contour
    )

    # ==================================
    # EXACTLY 4 POINTS
    # ==================================

    points = [

        # right
        (
            (cx + radius) / w,
            cy / h
        ),

        # bottom
        (
            cx / w,
            (cy + radius) / h
        ),

        # left
        (
            (cx - radius) / w,
            cy / h
        ),

        # top
        (
            cx / w,
            (cy - radius) / h
        )
    ]

    # ==================================
    # SAVE IMAGE
    # ==================================

    shutil.copy(
        frame_path,
        os.path.join(
            IMAGES_DIR,
            frame_name
        )
    )

    # ==================================
    # SAVE LABEL
    # ==================================

    label_path = os.path.join(
        LABELS_DIR,
        os.path.splitext(frame_name)[0]
        + ".txt"
    )

    with open(label_path, "w") as f:

        line = "0"

        for x, y in points:

            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            line += (
                f" {x:.6f}"
                f" {y:.6f}"
            )

        f.write(line + "\n")

    saved += 1

print("\n====================")
print("DONE")
print("====================")
print(f"Saved Images : {saved}")
print(f"Images Folder: {IMAGES_DIR}")
print(f"Labels Folder: {LABELS_DIR}")
print("====================")