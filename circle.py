import cv2
import numpy as np
import os

# ==================================
# PATHS
# ==================================

FRAME_DIR = "extracted_frames"
MASK_DIR = "masks"

os.makedirs(MASK_DIR, exist_ok=True)

frames = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# ==================================
# GLOBALS
# ==================================

mouse_x = 0
mouse_y = 0

radius = 20

current_idx = 0
total_frames = len(frames)

save_requested = False

# ==================================
# MOUSE CALLBACK
# ==================================

def mouse_callback(event, x, y, flags, param):

    global mouse_x, mouse_y
    global radius
    global save_requested

    mouse_x = x
    mouse_y = y

    # Left click = save current annotation
    if event == cv2.EVENT_LBUTTONDOWN:
        save_requested = True

    # Mouse wheel = resize circle
    elif event == cv2.EVENT_MOUSEWHEEL:

        if flags > 0:
            radius += 2
        else:
            radius = max(2, radius - 2)

# ==================================
# WINDOW
# ==================================

cv2.namedWindow("Ball Annotator", cv2.WINDOW_NORMAL)

cv2.setMouseCallback(
    "Ball Annotator",
    mouse_callback
)

# ==================================
# MAIN LOOP
# ==================================

while True:

    if current_idx < 0:
        current_idx = 0

    if current_idx >= total_frames:
        break

    frame_name = frames[current_idx]

    frame_path = os.path.join(
        FRAME_DIR,
        frame_name
    )

    img = cv2.imread(frame_path)

    if img is None:
        current_idx += 1
        continue

    while True:

        disp = img.copy()

        # ==================================
        # DRAW CIRCLE
        # ==================================

        cv2.circle(
            disp,
            (mouse_x, mouse_y),
            radius,
            (0, 255, 0),
            2
        )

        cv2.circle(
            disp,
            (mouse_x, mouse_y),
            3,
            (0, 0, 255),
            -1
        )

        # ==================================
        # INFO PANEL
        # ==================================

        cv2.rectangle(
            disp,
            (0, 0),
            (550, 180),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            disp,
            f"Frame: {current_idx+1}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            disp,
            f"Radius: {radius}",
            (10,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            disp,
            "Move Mouse = Move Circle",
            (10,95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1
        )

        cv2.putText(
            disp,
            "Wheel = Resize Circle",
            (10,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1
        )

        cv2.putText(
            disp,
            "LEFT CLICK = Save & Next Frame",
            (10,145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            1
        )

        cv2.putText(
            disp,
            "D=Next  A=Prev  ESC=Exit",
            (10,170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1
        )

        cv2.imshow(
            "Ball Annotator",
            disp
        )

        # ==================================
        # SAVE MASK ON MOUSE CLICK
        # ==================================

        if save_requested:

            # Create black RGB image
            mask = np.zeros(
                (img.shape[0], img.shape[1], 3),
                dtype=np.uint8
            )

            # Draw RED ball
            cv2.circle(
                mask,
                (mouse_x, mouse_y),
                radius,
                (0, 0, 255),   # RED in OpenCV (BGR)
                -1
            )

            mask_name = (
                os.path.splitext(frame_name)[0]
                + ".png"
            )

            cv2.imwrite(
                os.path.join(
                    MASK_DIR,
                    mask_name
                ),
                mask
            )

            print(
                f"Saved: {mask_name}"
            )

            save_requested = False

            current_idx += 1
            break
        key = cv2.waitKey(10) & 0xFF

        # ==================================
        # NEXT FRAME
        # ==================================

        if key == ord('d'):

            current_idx += 1
            break

        # ==================================
        # PREVIOUS FRAME
        # ==================================

        elif key == ord('a'):

            current_idx -= 1
            break

        # ==================================
        # INCREASE RADIUS
        # ==================================

        elif key == ord('+') or key == ord('='):

            radius += 2

        # ==================================
        # DECREASE RADIUS
        # ==================================

        elif key == ord('-'):

            radius = max(
                2,
                radius - 2
            )

        # ==================================
        # EXIT
        # ==================================

        elif key == 27:

            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

print("Annotation Finished")