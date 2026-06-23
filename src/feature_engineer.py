import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from src.utils import IMAGE_SIZE


class FeatureEngineer:
    def __init__(self, image_size=None):
        self.image_size = image_size or IMAGE_SIZE
        self.feature_names_ = None

    def extract_hog_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), feature_vector=True)
        return features

    def extract_color_histogram(self, image, bins=32):
        features = []
        for c in range(3):
            hist = cv2.calcHist([image], [c], None, [bins], [0, 256])
            hist = hist.flatten() / hist.sum()
            features.extend(hist)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for c in range(3):
            hist = cv2.calcHist([hsv], [c], None, [bins], [0, 256])
            hist = hist.flatten() / hist.sum()
            features.extend(hist)

        return np.array(features)

    def extract_edge_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        return np.array([
            edge_density,
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude)
        ])

    def extract_lbp_features(self, image, radius=3, n_points=24, bins=26):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        return hist

    def extract_color_moments(self, image):
        features = []
        for c in range(3):
            channel = image[:, :, c].astype(float)
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.cbrt(np.mean((channel - np.mean(channel)) ** 3)) if np.std(channel) > 0 else 0
            ])

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for c in range(3):
            channel = hsv[:, :, c].astype(float)
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])

        return np.array(features)

    def extract_features(self, image):
        hog_feat = self.extract_hog_features(image)
        color_hist = self.extract_color_histogram(image)
        edge_feat = self.extract_edge_features(image)
        lbp_feat = self.extract_lbp_features(image)
        color_moments = self.extract_color_moments(image)

        return np.concatenate([hog_feat, color_hist, edge_feat, lbp_feat, color_moments])

    def transform_dataset(self, images):
        print(f"Extracting features from {len(images)} images...")
        features = []
        for i, img in enumerate(images):
            feat = self.extract_features(img)
            features.append(feat)
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(images)} images")

        feature_matrix = np.array(features)
        self._build_feature_names(feature_matrix.shape[1])
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        return feature_matrix

    def _build_feature_names(self, total_features):
        names = []

        sample_img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        hog_len = len(self.extract_hog_features(sample_img))
        for i in range(hog_len):
            names.append(f"HOG_{i}")

        channels = ['R', 'G', 'B', 'H', 'S', 'V']
        for ch in channels:
            for i in range(32):
                names.append(f"ColorHist_{ch}_{i}")

        edge_names = ['EdgeDensity', 'EdgeMeanMag', 'EdgeStdMag', 'EdgeMaxMag']
        names.extend(edge_names)

        for i in range(26):
            names.append(f"LBP_{i}")

        for ch in ['R', 'G', 'B']:
            names.extend([f'{ch}_Mean', f'{ch}_Std', f'{ch}_Skew'])
        for ch in ['H', 'S', 'V']:
            names.extend([f'{ch}_Mean', f'{ch}_Std'])

        if len(names) < total_features:
            for i in range(total_features - len(names)):
                names.append(f"Extra_{i}")
        elif len(names) > total_features:
            names = names[:total_features]

        self.feature_names_ = names
        return names

    def get_feature_names(self):
        if self.feature_names_ is None:
            sample_img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            feat = self.extract_features(sample_img)
            self._build_feature_names(len(feat))
        return self.feature_names_
