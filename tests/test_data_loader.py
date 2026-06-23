import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from src.data_loader import DataLoader


@pytest.fixture
def sample_dataset(tmp_path):
    classes = ['cardboard', 'glass', 'metal']
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))
            img.save(cls_dir / f"{cls}{i}.jpg")
    return tmp_path, classes


class TestDataLoader:
    def test_load_dataset(self, sample_dataset):
        data_dir, classes = sample_dataset
        loader = DataLoader(data_dir=data_dir, image_size=(64, 64))
        images, labels = loader.load_dataset()
        assert len(images) == 15
        assert len(labels) == 15

    def test_image_shape(self, sample_dataset):
        data_dir, _ = sample_dataset
        loader = DataLoader(data_dir=data_dir, image_size=(64, 64))
        images, _ = loader.load_dataset()
        assert images.shape[1:] == (64, 64, 3)

    def test_label_values(self, sample_dataset):
        data_dir, classes = sample_dataset
        loader = DataLoader(data_dir=data_dir, image_size=(64, 64))
        _, labels = loader.load_dataset()
        unique = np.unique(labels)
        assert len(unique) <= len(classes)
        assert all(0 <= l < len(loader.class_names) for l in labels)

    def test_pixel_range(self, sample_dataset):
        data_dir, _ = sample_dataset
        loader = DataLoader(data_dir=data_dir, image_size=(64, 64))
        images, _ = loader.load_dataset()
        assert images.min() >= 0
        assert images.max() <= 255

    def test_get_class_names(self, sample_dataset):
        data_dir, _ = sample_dataset
        loader = DataLoader(data_dir=data_dir)
        names = loader.get_class_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_sample_images(self, sample_dataset):
        data_dir, _ = sample_dataset
        loader = DataLoader(data_dir=data_dir, image_size=(64, 64))
        loader.load_dataset()
        samples = loader.get_sample_images(n_per_class=2)
        assert isinstance(samples, dict)
        for key, imgs in samples.items():
            assert len(imgs) <= 2

    def test_empty_directory(self, tmp_path):
        loader = DataLoader(data_dir=tmp_path, image_size=(64, 64))
        images, labels = loader.load_dataset()
        assert len(images) == 0

    def test_corrupt_image_skipped(self, tmp_path):
        cls_dir = tmp_path / "cardboard"
        cls_dir.mkdir()
        corrupt_file = cls_dir / "corrupt.jpg"
        corrupt_file.write_text("not an image")
        img = Image.fromarray(np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))
        img.save(cls_dir / "valid.jpg")

        loader = DataLoader(data_dir=tmp_path, image_size=(64, 64))
        images, labels = loader.load_dataset()
        assert len(images) == 1
