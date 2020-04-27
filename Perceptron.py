# Perceptron
from scipy.sparse import csr_matrix
import numpy as np


class MlabPerceptron:
    "This class implements a Perceptron Classifier"

    def __init__(self, num_epochs=10, alpha=0.5):
        """
        Initialize the classfier
        :param num_epochs: how many epochs to run on the data
        :param alpha: learning rate
        """
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.w = None  # no weights

    def fit(self, X, y, verbose=False):
        """
        Train the classfier
        :param X: features
        :param y: labels
        """
        if isinstance(X, csr_matrix):
            X = X.todense()
        y = np.array(y, dtype=np.int)
        y[y == 0] = -1  # convert 0 -> -1
        num_samples = X.shape[0]
        num_features = X.shape[1]
        # initialize weights
        self.w = np.ones((1, num_features + 1))
        # train
        for epoch in range(self.num_epochs):
            num_updates = 0  # how many updates were performed

            for i_samp in range(num_samples):
                # concatenate 1 to feature vector
                sample = np.append(X[i_samp], np.ones((1, 1)), axis=1)
                prod = np.dot(self.w, np.transpose(sample))
                if prod < 0:
                    phy = -1
                else:
                    phy = 1
                if phy != y[i_samp]:
                    num_updates = num_updates + 1
                    self.w = self.w + self.alpha * (y[i_samp] - phy) * sample

            # end for i_samp
            if num_updates == 0:
                break

            if verbose:
                print("epoch {}: {} updates".format(epoch, num_updates))
        # end for epoch

    def predict(self, X):
        """
        Predict labels for features
        :param X: features
        :return y_pred: predictions
        """
        if self.w is None:
            print("can't call 'predict' before 'fit'")
            return
        num_samples = X.shape[0]
        num_features = X.shape[1]
        if isinstance(X, csr_matrix):
            X = X.todense()
        """
        Your Code Here
        """
        """ Starter code     """
        y_pred = []
        for i_samp in range(num_samples):
            # concatenate 1 to feature vector
            sample = np.append(X[i_samp], np.ones((1, 1)), axis=1)
            prod = np.dot(self.w, np.transpose(sample))
            if prod < 0:
                phy = -1
            else:
                phy = 1
            y_pred.append(phy)
        y_pred = np.array(y_pred)

        # end for i_samp
        y_pred[y_pred == -1] = 0  # convert back -1 -> 0
        return y_pred