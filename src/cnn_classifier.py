import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from src.utils import NUM_CLASSES, RANDOM_STATE


class CNNClassifier:
    def __init__(self, num_classes=None, learning_rate=0.001, epochs=10, batch_size=32):
        self.num_classes = num_classes or NUM_CLASSES
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._build_model()

    def _build_model(self):
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, self.num_classes)
        )
        self.model = self.model.to(self.device)

    def _prepare_data(self, images, labels=None):
        tensors = []
        for img in images:
            tensor = self.transform(img)
            tensors.append(tensor)
        X = torch.stack(tensors)

        if labels is not None:
            y = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=labels is not None)
        return loader

    def fit(self, images, labels):
        print(f"Training MobileNetV2 on {self.device}...")
        torch.manual_seed(RANDOM_STATE)

        train_loader = self._prepare_data(images, labels)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            acc = correct / total
            avg_loss = running_loss / len(train_loader)
            print(f"  Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")

        return self

    def predict(self, images):
        self.model.eval()
        test_loader = self._prepare_data(images)

        predictions = []
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, images):
        self.model.eval()
        test_loader = self._prepare_data(images)

        probabilities = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = softmax(outputs)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def score(self, images, labels):
        predictions = self.predict(images)
        return np.mean(predictions == labels)
