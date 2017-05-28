from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys

"""
    cnn.py
    training using neural networks
"""


def getDir(mode):
    return "./Data/Extracted_features/{0}.npy".format(mode)


def load(mode):
    f = open(getDir(mode))
    fake_x = np.load(f)
    f.close()

    return fake_x


class trainingModel:
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            # self.model = gaussian_process.GaussianProcessClassifier()
            self.model = MLPClassifier(hidden_layer_sizes=(100, 100), alpha=0.05)

    def train(self, X, Y):
        print (self.model.fit(X, Y))

    def classify(self, X):
        return self.model.predict(X)

    def validate(self, X, idx):
        data = self.model.predict_proba(X)[:, idx]
        return np.mean(self.model.predict(X)), np.mean(data), np.std(data)


def main(mode):
    # option
    print("mode: {0}".format(sys.argv[1]))

    # load
    features = load(mode)

    cla = np.array(np.loadtxt('Data/YelpZip/metadata',usecols=[3], dtype='string', delimiter='\t'))
                
    cla = (cla == '1')

    print("real: {0}\nfake: {1}\n".format(np.count_nonzero(cla), len(cla) - np.count_nonzero(cla)))

    print("test loaded")
    
    # cross validation
    splitter = StratifiedKFold(n_splits=3)
    for train_idx, test_idx in splitter.split(features, cla):
        X_train = features[train_idx]
        X_test = features[test_idx]
        Y_train = cla[train_idx]
        Y_test = cla[test_idx]

        # training model
        m = trainingModel()

        # train
        m.train(X_train, Y_train)

        print("trained")

        # partition
        fake_test = X_test[- Y_test]
        real_test = X_test[Y_test]

        # validation
        acc, prob, sd = m.validate(fake_test, 0)
        acc = 1 - acc
        print("Fake review (precision, prob, sd): {0}".format((acc, prob, sd)))
        print("Real review (precision, prob, sd): {0}".format(m.validate(real_test, 1)))


if __name__ == '__main__':
    main(sys.argv[1].strip())
