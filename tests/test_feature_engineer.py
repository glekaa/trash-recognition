import pytest
import numpy as np
from src.feature_engineer import FeatureEngineer


@pytest.fixture
def engineer():
    return FeatureEngineer(image_size=(128, 128))


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_images():
    return np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)


class TestFeatureEngineer:
    def test_extract_hog_features(self, engineer, sample_image):
        features = engineer.extract_hog_features(sample_image)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(np.isfinite(features))

    def test_extract_color_histogram(self, engineer, sample_image):
        features = engineer.extract_color_histogram(sample_image)
        assert isinstance(features, np.ndarray)
        assert len(features) == 32 * 6
        assert all(f >= 0 for f in features)

    def test_extract_edge_features(self, engineer, sample_image):
        features = engineer.extract_edge_features(sample_image)
        assert isinstance(features, np.ndarray)
        assert len(features) == 4
        assert 0 <= features[0] <= 1

    def test_extract_lbp_features(self, engineer, sample_image):
        features = engineer.extract_lbp_features(sample_image)
        assert isinstance(features, np.ndarray)
        assert len(features) == 26

    def test_extract_color_moments(self, engineer, sample_image):
        features = engineer.extract_color_moments(sample_image)
        assert isinstance(features, np.ndarray)
        assert len(features) == 15

    def test_extract_features_combined(self, engineer, sample_image):
        features = engineer.extract_features(sample_image)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(np.isfinite(features))

    def test_transform_dataset(self, engineer, sample_images):
        feature_matrix = engineer.transform_dataset(sample_images)
        assert feature_matrix.shape[0] == 10
        assert feature_matrix.shape[1] > 0

    def test_feature_consistency(self, engineer, sample_image):
        f1 = engineer.extract_features(sample_image)
        f2 = engineer.extract_features(sample_image)
        np.testing.assert_array_equal(f1, f2)

    def test_get_feature_names(self, engineer):
        names = engineer.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_match_features(self, engineer, sample_image):
        features = engineer.extract_features(sample_image)
        names = engineer.get_feature_names()
        assert len(features) == len(names)
