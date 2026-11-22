from pathlib import Path
import random
import shutil

from Python.config import (
    VALID_DIR,
    TEST_DIR,
    CLASS_NAMES,
    SUPPORTED_IMAGE_EXTENSIONS
)

TEST_IMAGES_PER_CLASS = 500
RANDOM_SEED = 42


def get_images(folder):
    return [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def main():
    print("WORKSHIELD TEST SPLIT")
    print("=" * 40)

    for index, class_name in enumerate(CLASS_NAMES):

        source_folder = VALID_DIR / class_name
        test_folder = TEST_DIR / class_name

        test_folder.mkdir(parents=True, exist_ok=True)

        existing_test_images = get_images(test_folder)

        if existing_test_images:
            raise RuntimeError(
                f"Test folder already contains images: {test_folder}"
            )

        images = get_images(source_folder)

        if len(images) <= TEST_IMAGES_PER_CLASS:
            raise ValueError(
                f"Not enough images in {source_folder}"
            )

        random_generator = random.Random(
            RANDOM_SEED + index
        )

        random_generator.shuffle(images)

        selected_images = images[:TEST_IMAGES_PER_CLASS]

        for image in selected_images:
            destination = test_folder / image.name

            shutil.move(
                str(image),
                str(destination)
            )

        print(f"\nClass: {class_name.upper()}")
        print(f"Moved to test: {len(selected_images)}")
        print(f"Remaining validation: {len(images) - len(selected_images)}")

    print("\nTest split completed successfully.")
    print("Training images were not changed.")


if __name__ == "__main__":
    main()