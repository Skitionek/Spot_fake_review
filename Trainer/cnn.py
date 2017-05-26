from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys

"""
    cnn.py
    training using neural networks
"""


def getDir(dirname, mode):
    return "../Data/Extracted_features/{0}/{1}.npy".format(dirname, mode)


def load(mode):
    f = open(getDir("Fake", mode))
    fake_x = np.load(f)
    f.close()

    f = open(getDir("notFake", mode))
    real_x = np.load(f)
    f.close()

    return fake_x, np.zeros(len(fake_x)), real_x, np.ones(len(real_x))


class trainingModel:
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            # self.model = gaussian_process.GaussianProcessClassifier()
            self.model = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1e-3, random_state=1)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def classify(self, X):
        return self.model.predict(X)

    def validate(self, X, Y):
        result = self.classify(X)
        # compare
        return float(np.count_nonzero(result == Y)) / float(len(Y))


def main(mode):
    # option
    print("mode: ", sys.argv[1])

    # load
    fake_x, fake_y, real_x, real_y = load(mode)

    # concatenate
    X = np.concatenate((fake_x, real_x), axis=0)
    Y = np.concatenate((fake_y, real_y), axis=0)

    print("test loaded")

    # tra
    m = trainingModel()
    # cross-validation?
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    print("cross validated")

    # train
    m.train(X_train, Y_train)

    print("trained")

    print("precision : ", m.validate(X_test, Y_test))
    return m


if __name__ == '__main__':
    main(sys.argv[1].strip())