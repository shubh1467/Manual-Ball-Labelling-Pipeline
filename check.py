import cv2
import numpy as np
import os

# =====================================
# PATHS
# =====================================

IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"

# =====================================
# FILES
# =====================================

images = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

print(
    f"Found {len(images)} images"
)

idx = 0

# =====================================
# LOOP
# =====================================

while True:

    if idx < 0:
        idx = 0

    if idx >= len(images):
        break

    image_name = images[idx]

    image_path = os.path.join(
        IMAGES_DIR,
        image_name
    )

    label_path = os.path.join(
        LABELS_DIR,
        os.path.splitext(image_name)[0]
        + ".txt"
    )

    image = cv2.imread(image_path)

    if image is None:
        idx += 1
        continue

    h, w = image.shape[:2]

    overlay = image.copy()

    # =====================================
    # READ LABEL
    # =====================================

    if os.path.exists(label_path):

        with open(label_path, "r") as f:

            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) != 9:
                continue

            coords = list(
                map(
                    float,
                    values[1:]
                )
            )

            points = []

            for i in range(
                0,
                len(coords),
                2
            ):

                x = int(
                    coords[i] * w
                )

                y = int(
                    coords[i + 1] * h
                )

                points.append(
                    [x, y]
                )

            points = np.array(
                points,
                dtype=np.int32
            )

            # =====================================
            # ORANGE FILLED MASK
            # =====================================

            cv2.fillPoly(
                overlay,
                [points],
                (0, 165, 255)   # Orange (BGR)
            )

    # =====================================
    # VISUALIZATION
    # =====================================

    vis = cv2.addWeighted(
        image,
        0.6,
        overlay,
        0.4,
        0
    )

    cv2.putText(
        vis,
        f"{idx+1}/{len(images)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        vis,
        image_name,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Ball Mask Check",
        vis
    )

    key = cv2.waitKey(0) & 0xFF

    # Next image
    if key == ord("d"):
        idx += 1

    # Previous image
    elif key == ord("a"):
        idx -= 1

    # Exit
    elif key == 27:
        break

cv2.destroyAllWindows()