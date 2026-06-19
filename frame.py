import cv2
import os
import sys

# ==================================
# PATHS
# ==================================

VIDEOS_DIR = "/Users/takneekmacmini/Documents/Ball Labelling Pipeline/data for shubham tiwari"

OUTPUT_DIR = "extracted_frames"

# ==================================
# SETTINGS
# ==================================

STRIDE = 1  # Save every frame (1 = all frames, 25 = every 25th frame)

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv",
    ".MP4", ".AVI", ".MOV", ".MKV"
)

# ==================================
# CREATE OUTPUT DIRECTORY
# ==================================

# Create full path
full_output_path = os.path.join(os.getcwd(), OUTPUT_DIR)

os.makedirs(full_output_path, exist_ok=True)

print(f"Output directory: {full_output_path}")
print(f"Current working directory: {os.getcwd()}")

# ==================================
# GET VIDEOS
# ==================================

if not os.path.exists(VIDEOS_DIR):
    print(f"❌ ERROR: Videos directory not found!")
    print(f"   Path: {VIDEOS_DIR}")
    sys.exit(1)

video_files = sorted([
    f for f in os.listdir(VIDEOS_DIR)
    if f.endswith(VIDEO_EXTENSIONS)
])

print(f"\nFound {len(video_files)} videos")

if len(video_files) == 0:
    print(f"❌ No video files found in: {VIDEOS_DIR}")
    sys.exit(1)

# List all videos found
for vf in video_files:
    print(f"   - {vf}")

total_saved = 0
all_video_stats = []

# ==================================
# PROCESS VIDEOS
# ==================================

for video_name in video_files:

    video_path = os.path.join(VIDEOS_DIR, video_name)
    video_prefix = os.path.splitext(video_name)[0]

    print(f"\n{'='*50}")
    print(f"Processing: {video_name}")
    print(f"Path: {video_path}")
    print(f"{'='*50}")

    # Check if video file exists and is readable
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        continue

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   FPS: {fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Stride: {STRIDE}")

    frame_idx = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % STRIDE == 0:

            # Create filename with video prefix
            image_name = f"{video_prefix}_frame_{frame_idx:06d}.png"
            output_path = os.path.join(full_output_path, image_name)

            success = cv2.imwrite(output_path, frame)

            if success:
                saved_count += 1
                total_saved += 1

                if saved_count % 100 == 0:
                    print(f"   Saved {saved_count} frames...")
            else:
                print(f"   ❌ Failed to save: {output_path}")

        frame_idx += 1

        # Progress indicator every 1000 frames
        if frame_idx % 1000 == 0:
            print(f"   Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    # Calculate expected frames
    expected_frames = (total_frames + STRIDE - 1) // STRIDE

    print(f"\n   ✅ Finished {video_name}")
    print(f"   Processed frames: {frame_idx}")
    print(f"   Expected frames: {expected_frames}")
    print(f"   Actually saved: {saved_count}")

    all_video_stats.append({
        'name': video_name,
        'saved': saved_count,
        'total_frames': frame_idx
    })

# ==================================
# VERIFY OUTPUT
# ==================================

print(f"\n{'='*50}")
print("VERIFYING OUTPUT FILES")
print(f"{'='*50}")

# List all saved files
all_saved_files = sorted(os.listdir(full_output_path))
print(f"Total files in output folder: {len(all_saved_files)}")

# Group by video prefix
video_groups = {}
for f in all_saved_files:
    prefix = f.split('_frame_')[0] if '_frame_' in f else f
    if prefix not in video_groups:
        video_groups[prefix] = []
    video_groups[prefix].append(f)

print("\nFrames per video:")
for prefix, files in video_groups.items():
    print(f"   {prefix}: {len(files)} frames")

# ==================================
# SUMMARY
# ==================================

print(f"\n{'='*50}")
print("FRAME EXTRACTION DONE")
print(f"{'='*50}")
print(f"Total Frames Saved: {total_saved}")
print(f"Output Folder: {full_output_path}")
print(f"Files in folder: {len(all_saved_files)}")

if total_saved != len(all_saved_files):
    print(f"⚠️ WARNING: Total saved ({total_saved}) doesn't match files in folder ({len(all_saved_files)})")

print(f"\n{'='*50}")
print("PER-VIDEO SUMMARY")
print(f"{'='*50}")

for stat in all_video_stats:
    print(f"{stat['name']}: {stat['saved']} frames saved")

print(f"\n✅ Done!")

# ==================================
# OPTIONAL: Show first few files
# ==================================

if len(all_saved_files) > 0:
    print(f"\nSample of saved files:")
    for f in all_saved_files[:10]:
        print(f"   - {f}")
    if len(all_saved_files) > 10:
        print(f"   ... and {len(all_saved_files) - 10} more")