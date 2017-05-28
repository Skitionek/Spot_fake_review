from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys

"""
    cnn.py
    training using neural networks
"""


def getDir(dirname, mode):
    return "./Data/Extracted_features/{0}/{1}.npy".format(dirname, mode)


def load(mode):
    f = open(getDir("Fake", mode))
    fake_x = np.load(f)
    f.close()

    f = open(getDir("notFake", mode))
    real_x = np.load(f)
    f.close()

    return fake_x, real_x


class trainingModel:
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            # self.model = gaussian_process.GaussianProcessClassifier()
            self.model = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=5e-3)

    def train(self, X, Y):
        print (self.model.fit(X, Y))

    def classify(self, X):
        return self.model.predict(X)

    def validate(self, X, Y):
        return self.model.score(X, Y)
        # compare
        # return float(np.count_nonzero(result == Y)) / float(len(Y))


def main(mode):
    # option
    print("mode: ", sys.argv[1])

    # load
    fake, real = load(mode)

    # cross validation
    fake_train, fake_test = train_test_split(fake, test_size=0.4, random_state=0)
    real_train, real_test = train_test_split(real, test_size=0.4, random_state=0)

    # concatenate
    X_train = np.concatenate((fake_train, real_train),
                             axis=0)
    Y_train = np.concatenate((np.zeros(len(fake_train)),
                              np.ones(len(real_train))), axis=0)

    X_test = np.concatenate((fake_test, real_test), axis=0)
    Y_test_fake = np.zeros(len(fake_test))
    Y_test_real = np.ones(len(real_test))
    Y_test = np.concatenate((Y_test_fake, Y_test_real), axis=0)

    print("test loaded")

    # training model
    m = trainingModel()

    # train
    m.train(X_train, Y_train)

    print("trained")

    # validate
    print("Overall precision : ", m.validate(X_test, Y_test))

    print("Fake review precision: ", m.validate(fake_test, Y_test_fake))
    print("Real review precision: ", m.validate(real_test, Y_test_real))

    return m


if __name__ == '__main__':
    main(sys.argv[1].strip())
