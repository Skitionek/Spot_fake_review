from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys
# from svm_behav import y_train



import csv



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
    features = load(mode)


    cla = np.array(np.loadtxt('Data/YelpZip/metadata',usecols=[3], dtype='string', delimiter='\t'))
                
    cla = (cla == '1')
    
    # cross validation
    X_train, X_test, Y_train, Y_test = train_test_split(features, cla, test_size=0.4, random_state=0)

    print("test loaded")

    # training model
    m = trainingModel()

    # train
    m.train(X_train, Y_train)

    print("trained")

    # validate
    print("Overall precision : ", m.validate(X_test, Y_test))

    # partition
    fake_test = X_test[- Y_test]
    Y_test_fake = np.array([False] * len(fake_test))
    real_test = X_test[Y_test]
    Y_test_real = np.array([True] * len(real_test))

    print("Fake review precision: ", m.validate(fake_test, Y_test_fake))
    print("Real review precision: ", m.validate(real_test, Y_test_real))

    return m


if __name__ == '__main__':
    main(sys.argv[1].strip())
