import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from src.utils import ensure_dir, OUTPUT_DIR, NUM_CLASSES


class ModelEvaluator:
    def __init__(self, class_names, output_dir=None):
        self.class_names = class_names
        self.output_dir = output_dir or str(OUTPUT_DIR / "evaluation")
        ensure_dir(self.output_dir)

    def compute_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

    def plot_confusion_matrix(self, y_true, y_pred, classifier_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f"Confusion Matrix - {classifier_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        safe_name = classifier_name.replace(' ', '_').lower()
        plt.savefig(f"{self.output_dir}/confusion_matrix_{safe_name}.png", dpi=150)
        plt.close()
        return cm

    def plot_roc_curves(self, y_true, y_proba, classifier_name):
        if y_proba is None:
            return

        n_classes = len(self.class_names)
        y_bin = label_binarize(y_true, classes=range(n_classes))

        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("husl", n_classes)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i],
                     label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {classifier_name}")
        plt.legend(loc='lower right')
        plt.tight_layout()
        safe_name = classifier_name.replace(' ', '_').lower()
        plt.savefig(f"{self.output_dir}/roc_curves_{safe_name}.png", dpi=150)
        plt.close()

    def plot_classifier_comparison(self, results):
        classifiers = list(results.keys())
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        x = np.arange(len(classifiers))
        width = 0.2

        fig, ax = plt.subplots(figsize=(14, 7))
        colors = sns.color_palette("husl", len(metrics))

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[clf]['metrics'][metric] for clf in classifiers]
            ax.bar(x + i * width, values, width, label=label, color=colors[i])

        ax.set_xlabel("Classifier")
        ax.set_ylabel("Score")
        ax.set_title("Classifier Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(classifiers, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/classifier_comparison.png", dpi=150)
        plt.close()

    def generate_report(self, results, y_test=None):
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 70)

        for clf_name, result in results.items():
            report_lines.append(f"\n{'─' * 70}")
            report_lines.append(f"Classifier: {clf_name}")
            report_lines.append(f"{'─' * 70}")

            metrics = result['metrics']
            report_lines.append(f"  Accuracy:             {metrics['accuracy']:.4f}")
            report_lines.append(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
            report_lines.append(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
            report_lines.append(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
            report_lines.append(f"  Precision (macro):    {metrics['precision_macro']:.4f}")
            report_lines.append(f"  Recall (macro):       {metrics['recall_macro']:.4f}")
            report_lines.append(f"  F1-Score (macro):     {metrics['f1_macro']:.4f}")

            if 'best_params' in result and result['best_params']:
                report_lines.append(f"\n  Best Parameters: {result['best_params']}")

            if 'classification_report' in result:
                report_lines.append(f"\n  Classification Report:")
                report_lines.append(result['classification_report'])

        report_lines.append(f"\n{'=' * 70}")
        best_clf = max(results.keys(), key=lambda k: results[k]['metrics']['f1_weighted'])
        report_lines.append(f"Best Classifier: {best_clf} "
                            f"(F1={results[best_clf]['metrics']['f1_weighted']:.4f})")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)

        with open(f"{self.output_dir}/evaluation_report.txt", 'w') as f:
            f.write(report_text)

        print(report_text)
        return report_text

    def evaluate(self, trainer, X_test=None, y_test=None):
        X = X_test if X_test is not None else trainer.X_test
        y = y_test if y_test is not None else trainer.y_test

        results = {}
        for clf_name in trainer.models:
            y_pred = trainer.predict(clf_name, X)
            y_proba = trainer.predict_proba(clf_name, X)

            metrics = self.compute_metrics(y, y_pred)
            cm = self.plot_confusion_matrix(y, y_pred, clf_name)
            self.plot_roc_curves(y, y_proba, clf_name)

            clf_report = classification_report(y, y_pred,
                                               target_names=self.class_names,
                                               zero_division=0)

            results[clf_name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': clf_report,
                'best_params': trainer.best_params.get(clf_name, {})
            }

        self.plot_classifier_comparison(results)
        self.generate_report(results, y)

        return results
