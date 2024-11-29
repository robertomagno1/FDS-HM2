import numpy as np
from libs.math import sigmoid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class LogisticRegression:
    def __init__(self, num_features : int):
        self.parameters = np.random.normal(0, 0.01, num_features)
        
    def predict(self, x:np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        ##############################
        # Compute predictions by taking the dot product of input data (x) and model parameters,
        # followed by applying the sigmoid function to squash values into the range [0, 1].
        preds = sigmoid(np.dot(x, self.parameters))
        ##############################
        return preds
    
    @staticmethod
    def likelihood(preds, y : np.array) -> np.array:
        """
        Function to compute the log likelihood of the model parameters according to data x and label y.

        Args:
            preds: the predicted labels.
            y: the label array.

        Returns:
            log_l: the log likelihood of the model parameters according to data x and label y.
        """
        ##############################
        # Compute the log-likelihood for binary classification using the cross-entropy formula.
        # A small value (1e-15) is added to avoid log(0) errors.
        log_l = np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
        ##############################
        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        ############################## 
        # Update parameters using gradient ascent: 
        # Increase parameters in the direction of the gradient scaled by the learning rate.
        self.parameters += lr * gradient
        ##############################
        pass
        
    @staticmethod
    def compute_gradient(x : np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """
        ##############################
        # Compute the gradient as the average error between predicted and true labels,
        # weighted by the input data.
        gradient = np.dot(x.T, (y - preds)) / x.shape[0]
        ##############################
        return gradient

