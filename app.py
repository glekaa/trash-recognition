import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import joblib

from src.utils import CLASSES, IMAGE_SIZE, OUTPUT_DIR
from src.feature_engineer import FeatureEngineer
from src.cnn_classifier import CNNClassifier

app = FastAPI(title="Trash Recognition API")

# Path definitions
scaler_path = OUTPUT_DIR / "scaler.joblib"
cnn_path = OUTPUT_DIR / "mobilenetv2.pth"

# Load scaler
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

# Load traditional models dynamically
traditional_models = {}
for name in ["random_forest", "svm", "knn", "gradient_boosting"]:
    model_path = OUTPUT_DIR / f"{name}_model.joblib"
    if model_path.exists():
        traditional_models[name] = joblib.load(model_path)

# Load PyTorch model if it exists
cnn = None
if cnn_path.exists():
    cnn = CNNClassifier()
    cnn.model.load_state_dict(torch.load(cnn_path, map_location=cnn.device))
    cnn.model.eval()


def preprocess_image(image_bytes, target_size):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("cnn", enum=["cnn", "random_forest", "svm", "knn", "gradient_boosting"])
):
    try:
        contents = await file.read()

        if model == "cnn":
            if cnn is None:
                return JSONResponse(status_code=500, content={"error": "CNN model weights not found."})

            img = preprocess_image(contents, (224, 224))
            tensor = cnn.transform(img).unsqueeze(0).to(cnn.device)

            with torch.no_grad():
                outputs = cnn.model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            pred_idx = np.argmax(probs)
            predicted_class = CLASSES[pred_idx]

            probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        else:
            clf = traditional_models.get(model)
            if clf is None or scaler is None:
                return JSONResponse(status_code=500, content={"error": f"{model} model or scaler not found."})

            img = preprocess_image(contents, IMAGE_SIZE)

            engineer = FeatureEngineer()
            features = engineer.extract_features(img).reshape(1, -1)
            features_scaled = scaler.transform(features)

            pred_idx = clf.predict(features_scaled)[0]
            predicted_class = CLASSES[pred_idx]

            probabilities = {}
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(features_scaled)[0]
                probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        return {
            "filename": file.filename,
            "model_used": model,
            "prediction": predicted_class,
            "probabilities": probabilities
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to process image: {str(e)}"})


@app.get("/health")
def health():
    models_loaded = {name: name in traditional_models for name in ["random_forest", "svm", "knn", "gradient_boosting"]}
    models_loaded["cnn"] = cnn is not None
    models_loaded["scaler"] = scaler is not None

    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }
