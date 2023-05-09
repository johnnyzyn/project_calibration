import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score

def accuracy(logits, labels):
    arg = logits.argmax(-1)
    return accuracy_score(arg, labels.flatten())


def BS(logits, labels):
    p = softmax(logits, axis=1)
    y = label_binarize(np.array(labels), classes=range(logits.shape[1]))
    return np.average(np.sum((p - y)**2, axis=1))

def NLL(logits, labels):
    p = softmax(logits, axis=1)
    return log_loss(labels, p)

# taken and modified from TODO
class TemperatureScaling():

    def __init__(self, temp=1, maxiter=50, solver="BFGS", loss='NLL'):
        """
        Initialize class
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
        self.loss = loss

    def _loss_fun(self, x, logits, true):
        scaled_l = self.predict(logits, x)
        if self.loss == 'BS':
            loss = BS(scaled_l, true)
        elif self.loss == 'NLL':
            loss = NLL(scaled_l, true)
        return loss

    # Find the temperature
    def fit(self, logits, true, verbose=False):
        """
        Trains the model and finds optimal temperature
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
        Returns:
            the results of optimizer after minimizing is finished.
        """

        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]

        if verbose:
            print("Temperature:", 1/self.temp)

        return opt

    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return logits/self.temp
        else:
            return logits/temp


def logistic_func(probs):
    """
    >>> probs = np.array([[0.1,0.9], [0.5,0.5]])
    >>> logs = logistic_func(probs)
    >>> from scipy.special import softmax
    >>> softmax(logs, axis=1)
    array([[0.1, 0.9],
           [0.5, 0.5]])
    """
    logits = np.zeros_like(probs)
    n = probs.shape[0] - 1
    logits[:, :-1] = (np.log(probs[:, :-1]).transpose() - np.log(probs[:, -1]).transpose()).transpose()
    return logits