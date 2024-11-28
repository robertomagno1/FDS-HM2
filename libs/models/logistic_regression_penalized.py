import numpy as np
from libs.models.logistic_regression import LogisticRegression

import sys
import os

# Aggiungi la directory radice del progetto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from libs.models.logistic_regression import LogisticRegression

class LogisticRegressionPenalized(LogisticRegression):
    def __init__(self, num_features: int, lambda_: float = 0.1):
        super().__init__(num_features)
        self.lambda_ = lambda_
    
    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        ##############################
        # Update the parameters (weights) with gradient ascent and regularization:
        # - lr * gradient: Standard update term to maximize log-likelihood.
        # - self.lambda_ * self.parameters: Regularization term penalizing large weights.
        #   This helps prevent overfitting by shrinking weights towards zero.
        self.parameters += lr * (gradient - (self.lambda_ * self.parameters))
        ##############################
        pass
    
