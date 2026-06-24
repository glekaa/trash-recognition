import sys
import argparse
import os
import cv2
import numpy as np
import joblib
import torch
from pathlib import Path
from src.utils import CLASSES, IMAGE_SIZE, OUTPUT_DIR
from src.feature_engineer import FeatureEngineer
from src.cnn_classifier import CNNClassifier


def predict_traditional(image_path, model_type):
    model_file = OUTPUT_DIR / f"{model_type}_model.joblib"
    scaler_file = OUTPUT_DIR / "scaler.joblib"

    if not model_file.exists() or not scaler_file.exists():
        print(f"Error: Trained {model_type} model or scaler not found in outputs/.")
        print("Please run the pipeline first to train the models: `uv run python main.py`")
        sys.exit(1)

    print(f"Loading {model_type} model and scaler...")
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Extracting features...")
    engineer = FeatureEngineer()
    features = engineer.extract_features(img).reshape(1, -1)

    print("Normalizing features...")
    features_scaled = scaler.transform(features)

    print("Predicting...")
    pred_idx = model.predict(features_scaled)[0]
    predicted_class = CLASSES[pred_idx]

    print("\n" + "=" * 40)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Model: {model_type.upper()}")
    print(f"Predicted Class: {predicted_class}")
    print("=" * 40)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        print("\nProbability distribution:")
        for cls, prob in zip(CLASSES, probs):
            print(f"  {cls:12s}: {prob * 100:.2f}%")


def predict_cnn(image_path):
    model_file = OUTPUT_DIR / "mobilenetv2.pth"

    if not model_file.exists():
        print("Error: Trained CNN model weights (mobilenetv2.pth) not found in outputs/.")
        print("Please run the pipeline first to train the CNN: `uv run python main.py`")
        sys.exit(1)

    print("Loading CNN model...")
    cnn = CNNClassifier()
    cnn.model.load_state_dict(torch.load(model_file, map_location=cnn.device))
    cnn.model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Preprocessing and predicting...")
    tensor = cnn.transform(img).unsqueeze(0).to(cnn.device)

    with torch.no_grad():
        outputs = cnn.model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    predicted_class = CLASSES[pred_idx]

    print("\n" + "=" * 40)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Model: CNN (MobileNetV2)")
    print(f"Predicted Class: {predicted_class}")
    print("=" * 40)

    print("\nProbability distribution:")
    for cls, prob in zip(CLASSES, probs):
        print(f"  {cls:12s}: {prob * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict trash class for a new image")
    parser.add_argument("image_path", type=str, help="Path to the image to classify")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "random_forest", "svm", "knn", "gradient_boosting"],
        help="Classifier model to use (default: cnn)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image path '{args.image_path}' does not exist.")
        sys.exit(1)

    if args.model == "cnn":
        predict_cnn(args.image_path)
    else:
        predict_traditional(args.image_path, args.model)


if __name__ == "__main__":
    main()
