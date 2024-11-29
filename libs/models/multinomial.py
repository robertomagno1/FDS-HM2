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
        self.parameters = np.random.normal(0, 1e-3, (num_features, num_classes))

    ## Define a function to compute the raw scores for each class.
    def predict(self, X: np.array) -> np.array:
        """
        Function to compute the raw scores for each sample and each class.

        Args:
            X: it's the input data matrix. The shape is (N, H)

        Returns:
            scores: it's the matrix containing raw scores for each sample and each class. The shape is (N, K)"""
        
        scores = np.dot(X, self.parameters)

        scores = softmax(scores)

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
        Function to compute the cross entropy loss from the predicted labels and the true labels.

        Args:
            preds: it's the matrix containing probability for each sample and each class. The shape is (N, K)
            y_onehot: it's the label array in encoded as one hot vector. The shape is (N, K)

        Returns:
            loss: The scalar that is the mean error for each sample.
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
        Function to update the weights in-place.

        Args:
            gradient: the jacobian of the cross entropy loss.
            lr: the learning rate.

        Returns:
            None
        """

        ## Update the parameters in-place by subtracting the gradient scaled by the learning rate.
        self.parameters -=  lr * gradient
        
    
    ## Define a static method to compute the gradient of the cross-entropy loss.
    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute gradient of the cross entropy loss with respect the parameters. 

        Args:
            x: it's the input data matrix. The shape is (N, H)
            y: it's the label array in encoded as one hot vector. The shape is (N, K)
            preds: it's the predicted labels. The shape is (N, K)

        Returns:
            jacobian: A matrix with the partial derivatives of the loss. The shape is (H, K)
        """
        ## Get the number of samples in the input data.
        N = x.shape[0]

        ## Compute the gradient (Jacobian matrix) as the average gradient over all samples.
        ## This is done by multiplying the transpose of the input matrix with the difference
        ## between the predicted probabilities and the true labels.
        jacobian =  np.dot(x.T, (preds - y)) / N

        return jacobian
