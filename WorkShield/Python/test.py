"""Command-line test for one facial image."""
import argparse
from Python.inference import predict_image

def main():
    parser = argparse.ArgumentParser(description="Classify one facial image with WorkShield.")
    parser.add_argument("image_path", help="Path to a JPG, JPEG, PNG, BMP or WEBP image.")
    args = parser.parse_args()
    result = predict_image(args.image_path)
    print("\n" + "=" * 55)
    print("                 WORKSHIELD ASSESSMENT")
    print("=" * 55)
    print(f"Image: {result['image_path']}")
    print(f"Classification: {result['predicted_class']}")
    print(f"FAKE model score: {result['fake_score'] * 100:.2f}%")
    print(f"REAL model score: {result['real_score'] * 100:.2f}%")
    print(f"Selected class score: {result['selected_score'] * 100:.2f}%")
    print(f"Decision threshold: {result['threshold']:.2f}")
    print(f"Inference time: {result['inference_seconds']:.4f} seconds")
    print("\nAssessment guidance:")
    print(result["review_note"])
    print("\nIMPORTANT: A model classification is not proof of image authenticity, identity or a live person's presence.")
    print("=" * 55)

if __name__ == "__main__":
    main()
