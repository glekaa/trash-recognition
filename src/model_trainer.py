import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.utils import RANDOM_STATE


class ModelTrainer:
    def __init__(self, random_state=None):
        self.random_state = random_state or RANDOM_STATE
        self.scaler = StandardScaler()
        self.models = {}
        self.best_params = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_classifiers(self):
        return {
            'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True,
                       random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                    random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,
                                                            random_state=self.random_state)
        }

    def get_param_grids(self):
        return {
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'poly']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            'Gradient Boosting': {
                'n_estimators': [100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }

    def train(self, classifier_name, X_train=None, y_train=None):
        X = X_train if X_train is not None else self.X_train
        y = y_train if y_train is not None else self.y_train

        classifiers = self.get_classifiers()
        if classifier_name not in classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        model = classifiers[classifier_name]
        print(f"Training {classifier_name}...")
        model.fit(X, y)
        self.models[classifier_name] = model

        train_acc = model.score(X, y)
        print(f"  Train accuracy: {train_acc:.4f}")

        return model

    def tune_hyperparameters(self, classifier_name, X_train=None, y_train=None):
        X = X_train if X_train is not None else self.X_train
        y = y_train if y_train is not None else self.y_train

        classifiers = self.get_classifiers()
        param_grids = self.get_param_grids()

        if classifier_name not in classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        print(f"Tuning {classifier_name} hyperparameters...")
        grid_search = GridSearchCV(
            classifiers[classifier_name],
            param_grids[classifier_name],
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X, y)

        self.models[classifier_name] = grid_search.best_estimator_
        self.best_params[classifier_name] = grid_search.best_params_

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_all(self, tune=False, X_train=None, y_train=None):
        classifiers = self.get_classifiers()
        for name in classifiers:
            if tune:
                self.tune_hyperparameters(name, X_train, y_train)
            else:
                self.train(name, X_train, y_train)
        return self.models

    def predict(self, classifier_name, X_test=None):
        X = X_test if X_test is not None else self.X_test

        if classifier_name not in self.models:
            raise ValueError(f"Model '{classifier_name}' not trained")

        return self.models[classifier_name].predict(X)

    def predict_proba(self, classifier_name, X_test=None):
        X = X_test if X_test is not None else self.X_test

        if classifier_name not in self.models:
            raise ValueError(f"Model '{classifier_name}' not trained")

        model = self.models[classifier_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None

    def cross_validate(self, classifier_name, X=None, y=None, cv=5):
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        classifiers = self.get_classifiers()
        if classifier_name not in classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        scores = cross_val_score(classifiers[classifier_name], X, y, cv=cv,
                                 scoring='f1_weighted', n_jobs=-1)

        print(f"{classifier_name} CV scores: {scores}")
        print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores
