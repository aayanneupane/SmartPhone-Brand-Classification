from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging

class SmartphoneClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True))
        ])
        self.classes = ['iphone', 'samsung', 'pixel']
        self.logger = logging.getLogger(__name__)

    def train(self, features, labels):
        """
        Train the classifier with extracted image features
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            self.pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = self.pipeline.score(X_train, y_train)
            test_score = self.pipeline.score(X_test, y_test)
            
            y_pred = self.pipeline.predict(X_test)
            
            self.logger.info(f"Training accuracy: {train_score:.3f}")
            self.logger.info(f"Test accuracy: {test_score:.3f}")
            self.logger.info("\nClassification Report:")
            self.logger.info(classification_report(y_test, y_pred))
            
            return train_score, test_score
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, features):
        """
        Predict the smartphone brand from image features
        """
        try:
            probabilities = self.pipeline.predict_proba(features.reshape(1, -1))
            prediction = self.pipeline.predict(features.reshape(1, -1))[0]
            
            confidence = np.max(probabilities)
            
            return {
                'brand': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.classes, probabilities[0])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise
