"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
    HW1 Problem 3 & 4 
    Student: Yuhuan Ma
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        expandm = np.copy(X)
        for i in range (1, degree):
            expandm = np.c_[expandm, np.power(X,i+1)]
        return expandm

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        # the length of X
        n = len(X)
		# expand X into an (n, degree) array
        matrixX = self.polyfeatures(X, self.degree)
        self.mean = np.mean(matrixX, axis=0)
        self.std = np.std(matrixX, axis=0)
		# standardize the data
        matrixX = (matrixX - self.mean)/self.std
		# add 1s column of the matrix
        matrixX  = np.c_[np.ones([n, 1]), matrixX]

        n,d = matrixX.shape
        reg_matrix = self.reg_lambda*np.eye(d)
        reg_matrix[0, 0] = 0
        self.weight = np.linalg.pinv(matrixX.T @ matrixX + reg_matrix).dot(matrixX.T) @ y

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)
        matrixX = self.polyfeatures(X, self.degree)
        matrixX = (matrixX - self.mean)/self.std
        matrixX = np.c_[np.ones([n,1]), matrixX]
        # return the predict
        return matrixX.dot(self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    diff = (a-b)**2
    MSE = np.mean(diff)
    return MSE


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    model = PolynomialRegression(degree, reg_lambda)
    for i in range(1,n):
        model.fit(Xtrain[0:i+1], Ytrain[0:i+1])
        # mean squared error for train
        pred_train = model.predict(Xtrain[0:i+1])
        errorTrain[i] = mean_squared_error(Ytrain[0:i+1],pred_train)
        # mean squared error for test
        pred_test = model.predict(Xtest)
        errorTest[i] = mean_squared_error(Ytest,pred_test)

    return errorTrain,errorTest


