import pytest
import numpy as np
from src.model_trainer import ModelTrainer


@pytest.fixture
def trainer():
    return ModelTrainer(random_state=42)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 4, 100)
    return X, y


class TestModelTrainer:
    def test_split_data(self, trainer, sample_data):
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_stratified(self, trainer, sample_data):
        X, y = sample_data
        _, _, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        train_dist = np.bincount(y_train) / len(y_train)
        test_dist = np.bincount(y_test) / len(y_test)
        np.testing.assert_allclose(train_dist, test_dist, atol=0.2)

    def test_train_svm(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        model = trainer.train('SVM')
        assert model is not None
        assert 'SVM' in trainer.models

    def test_train_random_forest(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        model = trainer.train('Random Forest')
        assert model is not None

    def test_train_knn(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        model = trainer.train('KNN')
        assert model is not None

    def test_train_invalid_classifier(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        with pytest.raises(ValueError):
            trainer.train('InvalidClassifier')

    def test_predict(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        trainer.train('SVM')
        predictions = trainer.predict('SVM')
        assert len(predictions) == len(trainer.X_test)
        assert all(p in range(4) for p in predictions)

    def test_predict_proba(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        trainer.train('SVM')
        proba = trainer.predict_proba('SVM')
        assert proba is not None
        assert proba.shape == (len(trainer.X_test), 4)

    def test_predict_untrained(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        with pytest.raises(ValueError):
            trainer.predict('SVM')

    def test_train_all(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        models = trainer.train_all()
        assert len(models) == 4

    def test_scaler_applied(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        assert abs(trainer.X_train.mean()) < 1.0
        assert abs(trainer.X_train.std() - 1.0) < 0.5

    def test_cross_validate(self, trainer, sample_data):
        X, y = sample_data
        trainer.split_data(X, y)
        scores = trainer.cross_validate('SVM')
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)
