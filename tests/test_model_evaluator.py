import pytest
import numpy as np
import os
from src.model_evaluator import ModelEvaluator


@pytest.fixture
def evaluator(tmp_path):
    return ModelEvaluator(
        class_names=['class_a', 'class_b', 'class_c'],
        output_dir=str(tmp_path)
    )


@pytest.fixture
def sample_predictions():
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 0, 1, 2, 2, 1, 0, 1, 2, 0])
    y_proba = np.random.dirichlet([1, 1, 1], size=10)
    return y_true, y_pred, y_proba


class TestModelEvaluator:
    def test_compute_metrics(self, evaluator, sample_predictions):
        y_true, y_pred, _ = sample_predictions
        metrics = evaluator.compute_metrics(y_true, y_pred)
        assert 'accuracy' in metrics
        assert 'f1_weighted' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_weighted'] <= 1

    def test_compute_metrics_keys(self, evaluator, sample_predictions):
        y_true, y_pred, _ = sample_predictions
        metrics = evaluator.compute_metrics(y_true, y_pred)
        expected_keys = ['accuracy', 'precision_weighted', 'recall_weighted',
                         'f1_weighted', 'precision_macro', 'recall_macro', 'f1_macro']
        for key in expected_keys:
            assert key in metrics

    def test_perfect_predictions(self, evaluator):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = evaluator.compute_metrics(y, y)
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_weighted'] == 1.0

    def test_plot_confusion_matrix(self, evaluator, sample_predictions, tmp_path):
        y_true, y_pred, _ = sample_predictions
        cm = evaluator.plot_confusion_matrix(y_true, y_pred, "TestClassifier")
        assert cm.shape == (3, 3)
        assert os.path.exists(f"{tmp_path}/confusion_matrix_testclassifier.png")

    def test_plot_roc_curves(self, evaluator, sample_predictions, tmp_path):
        y_true, _, y_proba = sample_predictions
        evaluator.plot_roc_curves(y_true, y_proba, "TestClassifier")
        assert os.path.exists(f"{tmp_path}/roc_curves_testclassifier.png")

    def test_plot_roc_curves_none_proba(self, evaluator, sample_predictions):
        y_true, _, _ = sample_predictions
        evaluator.plot_roc_curves(y_true, None, "TestClassifier")

    def test_generate_report(self, evaluator, sample_predictions, tmp_path):
        y_true, y_pred, _ = sample_predictions
        metrics = evaluator.compute_metrics(y_true, y_pred)
        results = {
            'TestClassifier': {
                'metrics': metrics,
                'classification_report': 'test report',
                'best_params': {}
            }
        }
        report = evaluator.generate_report(results)
        assert 'TestClassifier' in report
        assert os.path.exists(f"{tmp_path}/evaluation_report.txt")

    def test_plot_classifier_comparison(self, evaluator, sample_predictions, tmp_path):
        y_true, y_pred, _ = sample_predictions
        metrics = evaluator.compute_metrics(y_true, y_pred)
        results = {
            'Clf_A': {'metrics': metrics},
            'Clf_B': {'metrics': metrics}
        }
        evaluator.plot_classifier_comparison(results)
        assert os.path.exists(f"{tmp_path}/classifier_comparison.png")
