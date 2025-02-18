from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging

class SmartphoneClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, C=10.0, gamma='scale'))
        ])
        self.classes = ['iphone', 'samsung', 'pixel']
        self.logger = logging.getLogger(__name__)
        self.is_trained = False

    def validate_features(self, features, purpose='training'):
        """
        Validate feature array
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
        """
        Train the classifier with enhanced validation and evaluation
        """
        try:
            self.logger.info("Starting model training")

            # Validate inputs
            self.validate_features(features, 'training')
            if len(features) != len(labels):
                raise ValueError("Number of features and labels must match")

            if not all(label in self.classes for label in labels):
                raise ValueError(f"Labels must be one of {self.classes}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Perform cross-validation
            self.logger.info("Performing cross-validation")
            cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
            self.logger.info(f"Cross-validation scores: {cv_scores}")
            self.logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            # Train the model
            self.logger.info("Training final model")
            self.pipeline.fit(X_train, y_train)

            # Evaluate the model
            train_score = self.pipeline.score(X_train, y_train)
            test_score = self.pipeline.score(X_test, y_test)

            y_pred = self.pipeline.predict(X_test)

            self.logger.info(f"Training accuracy: {train_score:.3f}")
            self.logger.info(f"Test accuracy: {test_score:.3f}")
            self.logger.info("\nClassification Report:")
            self.logger.info(classification_report(y_test, y_pred))

            self.logger.info("\nConfusion Matrix:")
            self.logger.info(confusion_matrix(y_test, y_pred))

            self.is_trained = True
            return train_score, test_score

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, features):
        """
        Predict with enhanced validation
        """
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")

            self.validate_features(features, 'prediction')

            # Reshape if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # Get probability scores
            self.logger.debug("Computing prediction probabilities")
            probabilities = self.pipeline.predict_proba(features)[0]

            # Create probabilities dictionary
            prob_dict = {brand: float(prob) for brand, prob in zip(self.classes, probabilities)}

            # Get prediction and confidence
            max_prob_brand = max(prob_dict.items(), key=lambda x: x[1])
            prediction = max_prob_brand[0]
            confidence = max_prob_brand[1]

            self.logger.debug(f"Predicted brand: {prediction} with confidence: {confidence:.3f}")

            return {
                'brand': prediction,
                'confidence': confidence,
                'probabilities': prob_dict
            }

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise