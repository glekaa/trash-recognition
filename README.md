# Trash Recognition System

An end-to-end Machine Learning and Deep Learning system for classifying trash into 6 categories (cardboard, glass, metal, paper, plastic, trash) using the Garbage Classification dataset.

This project implements:
1. **Working Program**: Unified `main.py` entry point.
2. **OOP Design**: Structuring logic into distinct classes: `DataLoader`, `DataAnalyzer`, `FeatureEngineer`, `ModelTrainer`, `ModelEvaluator`, and `CNNClassifier`.
3. **Version Control**: Git-enabled project repository.
4. **Data Analysis**: Distribution plotting, statistical channel parameters, PCA feature space visualization, pixel density analysis.
5. **Feature Engineering**: Feature vectors incorporating Histograms of Oriented Gradients (HOG), Color Histograms (RGB + HSV), Edge Densities (Canny/Sobel), LBP textures, and statistical moments.
6. **Train/Test Splitting**: Stratified 80/20 division to preserve class representations.
7. **Model Training**: Support Vector Machine (SVM) as baseline.
8. **Alternative Classifiers**: Random Forest, KNN, Gradient Boosting, and a deep learning **MobileNetV2 CNN** with fine-tuning.
9. **Unit Testing**: Pytest suite for loaders, feature engineers, trainers, and evaluators.
10. **Evaluation Metrics**: Comprehensive classification reports, confusion matrices, and ROC curves saved automatically.

---

## Installation & Setup

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd trash-recognition
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

---

## Running the Project

### 1. Run Pipeline End-to-End
Run the main script to perform data analysis, feature extraction, train traditional ML models, and train the MobileNetV2 CNN:
```bash
uv run python main.py
```

Arguments:
- `--skip-analysis`: Toggles off preliminary plotting.
- `--skip-cnn`: Skips neural network training (runs only traditional classifiers).
- `--tune`: Performs hyperparameter optimization using GridSearchCV.
- `--test-size`: Size of the test set (default `0.2`).

### 2. Classify a New Image (Inference)
Once the pipeline has completed training, you can classify any new, individual image using `predict.py`.

Use the CNN Model (Default):
```bash
uv run python predict.py /path/to/your/image.jpg
```

Use a Traditional Model (e.g. Random Forest or SVM):
```bash
uv run python predict.py /path/to/your/image.jpg --model random_forest
uv run python predict.py /path/to/your/image.jpg --model svm
```

### 3. Deploy prediction API (FastAPI)
You can deploy a lightweight HTTP API to serve predictions.

Start the FastAPI local development server:
```bash
uv run uvicorn app:app --reload
```

Use `curl` to send a POST request with an image file:
```bash
curl -X POST "http://127.0.0.1:8000/predict?model=cnn" -F "file=@sample-glass.jpg"
```

### 4. Run Unit Tests
```bash
uv run pytest
```



---

## Architecture Overview

- **`DataLoader`** ([data_loader.py](file:///Users/pm/Documents/University/trash-recognition/src/data_loader.py)): Handles reading classes, resizing, colorspace conversions (BGR to RGB), and data formatting.
- **`DataAnalyzer`** ([data_analyzer.py](file:///Users/pm/Documents/University/trash-recognition/src/data_analyzer.py)): Handles generating Exploratory Data Analysis (EDA) visualizations (class imbalance, statistics, correlation matrix, PCA projection).
- **`FeatureEngineer`** ([feature_engineer.py](file:///Users/pm/Documents/University/trash-recognition/src/feature_engineer.py)): Handles engineering high-quality features from images (HOG, Histograms, LBP, Moments, Canny Edge).
- **`ModelTrainer`** ([model_trainer.py](file:///Users/pm/Documents/University/trash-recognition/src/model_trainer.py)): Splits, scales, cross-validates, and trains ML classifiers (SVM, RF, KNN, Gradient Boosting).
- **`ModelEvaluator`** ([model_evaluator.py](file:///Users/pm/Documents/University/trash-recognition/src/model_evaluator.py)): Computes metrics, outputs classification reports, draws confusion heatmaps, and draws ROC/AUC plots.
- **`CNNClassifier`** ([cnn_classifier.py](file:///Users/pm/Documents/University/trash-recognition/src/cnn_classifier.py)): Fine-tunes a PyTorch pre-trained MobileNetV2 CNN classifier.

---

## Evaluation Results

Saved to `outputs/evaluation/evaluation_report.txt`:

| Classifier | Accuracy | Precision (weighted) | Recall (weighted) | F1-Score (weighted) |
|---|---|---|---|---|
| **MobileNetV2** | **81.62%** | **82.15%** | **81.62%** | **81.52%** |
| **Random Forest** | 77.67% | 77.99% | 77.67% | 77.64% |
| **Gradient Boosting** | 77.27% | 77.67% | 77.27% | 77.36% |
| **SVM** | 76.28% | 77.18% | 76.28% | 76.37% |
| **KNN** | 62.25% | 64.45% | 62.25% | 61.25% |
