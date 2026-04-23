import shutil
import random
from pathlib import Path

# Reproducibility
random.seed(42)

# Base paths
base_path = Path("artifacts/data_ingestion")
train_dir = base_path / "train"
val_dir = base_path / "val"

classes = ["Normal", "Tumor"]
split_ratio = 0.2  # 20% validation

# ✅ SAFETY CHECK (prevents accidental re-run)
if val_dir.exists() and any(val_dir.iterdir()):
    print("⚠️ Validation folder already exists and is not empty.")
    print("👉 Delete 'val' folder if you want to re-split.")
    exit()

# Split process
for cls in classes:
    class_train_path = train_dir / cls
    class_val_path = val_dir / cls

    # Create validation folder
    class_val_path.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = list(class_train_path.glob("*"))
    random.shuffle(images)

    # Calculate split
    n_val = int(len(images) * split_ratio)
    val_images = images[:n_val]

    # Move files
    for img in val_images:
        shutil.move(str(img), class_val_path / img.name)

    print(f"Moved {n_val} images from '{cls}' → val")

print("\n✅ Validation dataset created successfully!")