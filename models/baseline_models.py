"""
Baseline Machine Learning Model Module
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os


class BaselineDetector:
    """
    Baseline Anomaly Detector
    """
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize detector
        Args:
            model_type: Model type
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.is_fitted = False
        
    def _create_model(self):
        """Create model instance"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=0.5,
                gamma='auto',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=10.0,
                max_iter=2000,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=3,
                weights='distance',
                n_jobs=-1
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True
            )
        elif self.model_type == 'isolation_forest':
            return IsolationForest(
                n_estimators=200,
                contamination='auto',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _validate_data(self, X, y=None):
        """
        Validate and clean data before training
        """

        X = np.array(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isinf(X).any():
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
        
        if y is not None:
            y = np.array(y, dtype=np.int32)
            return X, y
        
        return X
    
    def train(self, X_train, y_train):
        """
        Train model
        """
        print(f"\nTraining {self.model_type} model...")

        X_train, y_train = self._validate_data(X_train, y_train)

        if np.isnan(X_train).any():
            print("Warning: Still found NaN in training data. Filling with 0.")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training data range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        
        if self.model_type == 'isolation_forest':
            normal_data = X_train[y_train == 0]
            if len(normal_data) == 0:
                raise ValueError("No normal data for Isolation Forest training!")
            self.model.fit(normal_data)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        print(f"Training completed.")
        return self
    
    def predict(self, X):
        """
        Predict labels for test data
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        X = self._validate_data(X)
        
        if self.model_type == 'isolation_forest':
            pred = self.model.predict(X)
            return (pred == -1).astype(int)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities for test data
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        X = self._validate_data(X)
        
        if self.model_type == 'isolation_forest':
            scores = self.model.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            probas = np.column_stack([1 - scores, scores])
            return probas
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            scores = self.model.decision_function(X)
            if len(scores.shape) == 1:
                probas = np.column_stack([1 - scores, scores])
            else:
                probas = scores
            return probas
    
    def save(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.model_type = data['model_type']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")
        return self
