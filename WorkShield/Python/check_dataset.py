"""Check WorkShield dataset folders and image counts."""
from Python.config import TRAIN_DIR, VALID_DIR, TEST_DIR, CLASS_NAMES
from Python.data_utils import count_class_images, validate_dataset

def show_split(name, folder):
    print(f"\n{name}: {folder}")
    if not folder.exists():
        print("  NOT FOUND"); return
    counts = count_class_images(folder)
    for class_name in CLASS_NAMES:
        print(f"  {class_name.upper():5}: {counts[class_name]} images")
    print(f"  TOTAL: {sum(counts.values())} images")

def main():
    print("WORKSHIELD DATASET CHECK")
    show_split("TRAINING", TRAIN_DIR)
    show_split("VALIDATION", VALID_DIR)
    show_split("TEST (OPTIONAL)", TEST_DIR)
    errors = validate_dataset()
    if errors:
        print("\nDataset problems:")
        for error in errors: print("-", error)
        raise SystemExit(1)
    print("\nTraining and validation folders are ready.")
    if not TEST_DIR.exists():
        print("No independent test set found. Evaluation will use validation data until a separate test set is supplied.")

if __name__ == "__main__":
    main()
