from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging
from joblib import parallel_backend
import time
import psutil
from datetime import datetime

class SmartphoneClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                C=10.0,
                gamma='scale',
                cache_size=1000,
                decision_function_shape='ovr',
                random_state=42
            ))
        ])
        self.classes = ['iphone', 'samsung', 'pixel']
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.label_encoder = LabelEncoder()

    def log_memory_usage(self):
        """Log current memory usage"""
        process = psutil.Process()
        mem_info = process.memory_info()
        self.logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")

    def validate_features(self, features, purpose='training'):
        """
        Validate feature array with detailed checks
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"Features must be a numpy array for {purpose}")

        if features.size == 0:
            raise ValueError(f"Empty feature array provided for {purpose}")

        if np.isnan(features).any():
            raise ValueError(f"Features contain NaN values in {purpose}")

        if np.isinf(features).any():
            raise ValueError(f"Features contain infinite values in {purpose}")

    def train(self, features, labels):
        """Train the classifier with enhanced progress logging"""
        try:
            start_time = time.time()
            self.logger.info(f"Starting model training at {datetime.now().strftime('%H:%M:%S')}")
            self.log_memory_usage()

            # Validate inputs
            self.validate_features(features, 'training')
            if len(features) != len(labels):
                raise ValueError("Number of features and labels must match")

            if not all(label in self.classes for label in labels):
                raise ValueError(f"Labels must be one of {self.classes}")

            # Convert string labels to numerical values
            numerical_labels = self.label_encoder.fit_transform(labels)

            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                features, numerical_labels, test_size=0.2, random_state=42,
                stratify=numerical_labels
            )
            self.logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples")

            # Perform cross-validation with parallel processing
            self.logger.info("\nStarting cross-validation phase...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []

            # Track each fold separately
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
                fold_start = time.time()
                self.logger.info(f"\nProcessing fold {fold}/3 at {datetime.now().strftime('%H:%M:%S')}")

                with parallel_backend('threading', n_jobs=2):
                    score = cross_val_score(
                        self.pipeline, 
                        X_train[train_idx], y_train[train_idx],
                        cv=[(slice(None), slice(None))],  # Single split for this fold
                        n_jobs=2
                    )[0]

                cv_scores.append(score)
                self.logger.info(f"Fold {fold} completed in {time.time() - fold_start:.1f}s, Score: {score:.3f}")
                self.log_memory_usage()

            self.logger.info(f"\nCross-validation scores: {cv_scores}")
            self.logger.info(f"Mean CV score: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")

            # Train final model
            self.logger.info("\nTraining final model...")
            with parallel_backend('threading', n_jobs=2):
                self.pipeline.fit(X_train, y_train)

            # Evaluate model
            train_score = self.pipeline.score(X_train, y_train)
            test_score = self.pipeline.score(X_test, y_test)
            y_pred = self.pipeline.predict(X_test)

            # Convert numerical predictions back to string labels for reporting
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)

            total_time = time.time() - start_time
            self.logger.info(f"\nTraining completed in {total_time:.1f} seconds")
            self.logger.info(f"Training accuracy: {train_score:.3f}")
            self.logger.info(f"Test accuracy: {test_score:.3f}")
            self.logger.info("\nClassification Report:")
            self.logger.info(classification_report(y_test_labels, y_pred_labels))
            self.logger.info("\nConfusion Matrix:")
            self.logger.info(confusion_matrix(y_test_labels, y_pred_labels))

            self.is_trained = True
            return train_score, test_score

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, features):
        """
        Predict with enhanced validation and parallel processing
        """
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")

            self.validate_features(features, 'prediction')

            # Reshape if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # Get probability scores using parallel processing
            self.logger.debug("Computing prediction probabilities")
            with parallel_backend('threading', n_jobs=-1):
                probabilities = self.pipeline.predict_proba(features)[0]

            # Get numerical prediction and convert back to label
            numerical_prediction = self.pipeline.predict(features)[0]
            prediction = self.label_encoder.inverse_transform([numerical_prediction])[0]

            # Create probabilities dictionary with original labels
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }

            # Get confidence from probabilities
            confidence = float(probabilities.max())

            self.logger.debug(f"Predicted brand: {prediction} with confidence: {confidence:.3f}")

            return {
                'brand': prediction,
                'confidence': confidence,
                'probabilities': prob_dict
            }

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise