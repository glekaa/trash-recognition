import sys
import argparse
import warnings
import numpy as np
from src.utils import ensure_dir, OUTPUT_DIR
from src.data_loader import DataLoader
from src.data_analyzer import DataAnalyzer
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator


def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Trash Recognition Pipeline")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip data analysis")
    parser.add_argument('--skip-cnn', action='store_true', help="Skip CNN training")
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning")
    parser.add_argument('--test-size', type=float, default=0.2, help="Test set ratio")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    print("=" * 60)
    print("TRASH RECOGNITION PIPELINE")
    print("=" * 60)

    print("\n[1/6] Loading dataset...")
    loader = DataLoader()
    images, labels = loader.load_dataset()
    if len(images) == 0:
        print("Error: No images loaded. Please check the dataset path.")
        sys.exit(1)
    print(f"  Loaded {len(images)} images across {len(loader.get_class_names())} classes")

    print("\n[2/6] Extracting features...")
    engineer = FeatureEngineer()
    features = engineer.transform_dataset(images)
    feature_names = engineer.get_feature_names()

    if not args.skip_analysis:
        print("\n[3/6] Running data analysis...")
        analyzer = DataAnalyzer(images, labels, loader.get_class_names())
        analyzer.run_full_analysis(features, feature_names)
    else:
        print("\n[3/6] Skipping data analysis")

    print("\n[4/6] Preparing training and test sets...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(features, labels, test_size=args.test_size)

    print("\n[5/6] Training classifiers...")
    if args.tune:
        print("  Hyperparameter tuning enabled (this may take a while)...")
        trainer.train_all(tune=True)
    else:
        trainer.train_all(tune=False)

    print("\n[6/6] Evaluating models...")
    evaluator = ModelEvaluator(loader.get_class_names())
    results = evaluator.evaluate(trainer)

    import joblib
    joblib.dump(trainer.scaler, OUTPUT_DIR / "scaler.joblib")
    if 'Random Forest' in trainer.models:
        joblib.dump(trainer.models['Random Forest'], OUTPUT_DIR / "random_forest_model.joblib")
    if 'SVM' in trainer.models:
        joblib.dump(trainer.models['SVM'], OUTPUT_DIR / "svm_model.joblib")

    if not args.skip_cnn:
        print("\n[BONUS] Training CNN (MobileNetV2)...")
        try:
            import torch
            from src.cnn_classifier import CNNClassifier
            from sklearn.model_selection import train_test_split
            from src.utils import RANDOM_STATE

            train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
                images, labels, test_size=args.test_size,
                random_state=RANDOM_STATE, stratify=labels
            )

            cnn = CNNClassifier(epochs=5, batch_size=32)
            cnn.fit(train_imgs, train_lbls)

            # Save PyTorch Model
            torch.save(cnn.model.state_dict(), OUTPUT_DIR / "mobilenetv2.pth")

            y_pred_cnn = cnn.predict(test_imgs)
            y_proba_cnn = cnn.predict_proba(test_imgs)

            cnn_metrics = evaluator.compute_metrics(test_lbls, y_pred_cnn)
            evaluator.plot_confusion_matrix(test_lbls, y_pred_cnn, "MobileNetV2")
            evaluator.plot_roc_curves(test_lbls, y_proba_cnn, "MobileNetV2")

            from sklearn.metrics import classification_report
            clf_report = classification_report(test_lbls, y_pred_cnn,
                                               target_names=loader.get_class_names(),
                                               zero_division=0)

            results["MobileNetV2"] = {
                'metrics': cnn_metrics,
                'confusion_matrix': None,
                'classification_report': clf_report,
                'best_params': {'epochs': 5, 'lr': 0.001, 'batch_size': 32}
            }

            evaluator.plot_classifier_comparison(results)
            evaluator.generate_report(results)

        except ImportError as e:
            print(f"  Skipping CNN: {e}")
            print("  Install torch and torchvision to enable CNN training")

    print("\nPipeline complete!")
    print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

