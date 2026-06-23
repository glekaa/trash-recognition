import numpy as np
import cv2
from pathlib import Path
from src.utils import IMAGE_SIZE, CLASSES, DATA_DIR


class DataLoader:
    def __init__(self, data_dir=None, image_size=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.image_size = image_size or IMAGE_SIZE
        self.images = None
        self.labels = None
        self.class_names = CLASSES

    def load_dataset(self):
        images = []
        labels = []
        for idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, self.image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(idx)
        self.images = np.array(images) if images else np.array([])
        self.labels = np.array(labels) if labels else np.array([])
        return self.images, self.labels

    def get_class_names(self):
        return self.class_names

    def get_sample_images(self, n_per_class=5):
        samples = {}
        for idx, class_name in enumerate(self.class_names):
            mask = self.labels == idx
            class_images = self.images[mask]
            n = min(n_per_class, len(class_images))
            samples[class_name] = class_images[:n]
        return samples
