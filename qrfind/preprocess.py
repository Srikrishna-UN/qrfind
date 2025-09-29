import os
import random
import shutil

# --- Configuration ---
# The script assumes your data is already in the 'train' folders.
IMAGE_SRC_DIR = "data/images/train"
LABEL_SRC_DIR = "data/labels/train"

# Define the destination directories for the new validation set.
IMAGE_DEST_DIR = "data/images/val"
LABEL_DEST_DIR = "data/labels/val"

# Define the proportion of the data to be used for validation (e.g., 0.2 for 20%)
VALIDATION_SPLIT = 0.2
# --- End of Configuration ---

def split_dataset():
    """
    Splits the dataset into training and validation sets.
    """
    # Create destination directories if they don't exist
    os.makedirs(IMAGE_DEST_DIR, exist_ok=True)
    os.makedirs(LABEL_DEST_DIR, exist_ok=True)
    print(f"Created validation directories: '{IMAGE_DEST_DIR}' and '{LABEL_DEST_DIR}'")

    # Get a list of all image files in the source directory
    all_images = [f for f in os.listdir(IMAGE_SRC_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    # Determine the number of files to move
    num_val_files = int(len(all_images) * VALIDATION_SPLIT)
    val_files = all_images[:num_val_files]

    if not val_files:
        print("Warning: No files to move for validation set. The dataset might be too small or split is 0.")
        return

    print(f"Total images: {len(all_images)}")
    print(f"Moving {len(val_files)} images (and their labels) to the validation set...")

    # Move the selected files and their corresponding labels
    moved_count = 0
    for image_filename in val_files:
        base_filename = os.path.splitext(image_filename)[0]
        label_filename = f"{base_filename}.txt"

        # Source paths
        image_src_path = os.path.join(IMAGE_SRC_DIR, image_filename)
        label_src_path = os.path.join(LABEL_SRC_DIR, label_filename)

        # Destination paths
        image_dest_path = os.path.join(IMAGE_DEST_DIR, image_filename)
        label_dest_path = os.path.join(LABEL_DEST_DIR, label_filename)

        # Check if both image and label exist before moving
        if os.path.exists(image_src_path) and os.path.exists(label_src_path):
            shutil.move(image_src_path, image_dest_path)
            shutil.move(label_src_path, label_dest_path)
            moved_count += 1
        else:
            print(f"Warning: Could not find matching label for '{image_filename}'. Skipping.")

    print(f"\nSplit complete. Moved {moved_count} image/label pairs.")
    print("Your dataset is now ready for training.")

if __name__ == "__main__":
    split_dataset()
