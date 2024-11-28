from libs.models.logistic_regression import LogisticRegression
import numpy as np
from libs.math import softmax

import sys
import os

# Aggiungi la directory radice del progetto al percorso
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))




## Define a SoftmaxClassifier class that inherits from LogisticRegression.
class SoftmaxClassifier(LogisticRegression):
    def __init__(self, num_features: int, num_classes: int):
        ## Initialize the model parameters with small random values.
        ## The parameters matrix has a shape of (num_features, num_classes), 
        ## where each column corresponds to the weights for a specific class.
        self.parameters = np.random.normal(0, 1e-3, (num_features, num_classes))

    ## Define a function to compute the raw scores for each class.
    def predict(self, X: np.array) -> np.array:
       
        ## Compute the raw scores by performing a dot product between the input data 
        ## and the parameters matrix.
        scores = np.dot(X, self.parameters)

        return scores
    
    ## Define a function to compute the predicted class for each sample.
    def predict_labels(self, X: np.array) -> np.array:
        """
        Args:
            X: Input data matrix of shape (N, H).
            
        Returns:
            preds: Predicted class for each sample, with shape (N,).
        """

        ## Compute the raw scores using the predict function.
        scores = self.predict(X)

        ## Convert raw scores into probabilities using the softmax function.
        probabilities = softmax(scores)

        ## Take the class with the highest probability as the prediction.
        preds = np.argmax(probabilities, axis=1)

        return preds
    
    ## Define a static method to compute the cross-entropy loss.
    @staticmethod
    def likelihood(preds: np.array, y_onehot: np.array) -> float:
        """
        Args:
            preds: Matrix of probabilities for each sample and class, with shape (N, K).
            y_onehot: True labels encoded as one-hot vectors, with shape (N, K).

        Returns:
            loss: Average cross-entropy loss for all samples.
        """

        ## Clip probabilities to avoid numerical instability (e.g., log(0)).
        preds = np.clip(preds, 1e-15, 1 - 1e-15)

        ## Compute the cross-entropy loss for each sample.
        cross_entropy = -np.sum(y_onehot * np.log(preds), axis=1)

        ## Return the mean loss across all samples.
        loss = np.mean(cross_entropy)

        return loss
    
    ## Define a function to update the parameters using the computed gradient.
    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Args:
            gradient: Gradient of the loss with respect to the parameters.
            lr: Learning rate.

        Returns:
            None
        """

        ## Update the parameters in-place by subtracting the gradient scaled by the learning rate.
        self.parameters -= lr * gradient
    
    ## Define a static method to compute the gradient of the cross-entropy loss.
    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Args:
            x: Input data matrix of shape (N, H).
            y: True labels encoded as one-hot vectors, with shape (N, K).
            preds: Predicted probabilities, with shape (N, K).

        Returns:
            jacobian: Gradient matrix with shape (H, K).
        """

        ## Get the number of samples in the input data.
        N = x.shape[0]

        ## Compute the gradient (Jacobian matrix) as the average gradient over all samples.
        ## This is done by multiplying the transpose of the input matrix with the difference
        ## between the predicted probabilities and the true labels.
        jacobian = np.dot(x.T, (preds - y)) / N

        return jacobian
