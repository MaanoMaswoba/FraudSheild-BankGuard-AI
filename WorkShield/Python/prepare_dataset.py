"""Create stratified train/valid/test folders from Data/raw/fake and Data/raw/real."""
import argparse
import shutil
from Python.config import RAW_DATA_DIR, DATASET_DIR, CLASS_NAMES, SUPPORTED_IMAGE_EXTENSIONS
from sklearn.model_selection import train_test_split

def list_images(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)

def split_files(files, seed):
    if len(files) < 4:
        raise ValueError("Each class needs at least four images for a 70/15/15 split.")
    train_files, remaining = train_test_split(files, test_size=0.30, random_state=seed, shuffle=True)
    valid_files, test_files = train_test_split(remaining, test_size=0.50, random_state=seed, shuffle=True)
    return {"train": train_files, "valid": valid_files, "test": test_files}

def main():
    parser = argparse.ArgumentParser(description="Create a stratified 70/15/15 dataset split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated dataset.")
    args = parser.parse_args()
    class_files = {}
    for name in CLASS_NAMES:
        source = RAW_DATA_DIR / name
        if not source.exists(): raise FileNotFoundError(f"Missing raw class folder: {source}")
        class_files[name] = list_images(source)
        if not class_files[name]: raise ValueError(f"No supported images found in {source}")
    if DATASET_DIR.exists() and list_images(DATASET_DIR):
        if not args.overwrite:
            raise FileExistsError(f"{DATASET_DIR} already contains images. Use --overwrite only if you intend to replace them.")
        shutil.rmtree(DATASET_DIR)
    totals = {"train": 0, "valid": 0, "test": 0}
    for class_name, images in class_files.items():
        partitions = split_files(images, args.seed)
        for split, paths in partitions.items():
            destination = DATASET_DIR / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for image_path in paths:
                target = destination / image_path.name
                if target.exists(): target = destination / f"{image_path.stem}_{totals[split]}{image_path.suffix}"
                shutil.copy2(image_path, target)
                totals[split] += 1
    print("\nDataset preparation completed.")
    for split, count in totals.items(): print(f"{split.upper()}: {count} images")
    print("Keep the test split untouched until final evaluation.")

if __name__ == "__main__":
    main()
