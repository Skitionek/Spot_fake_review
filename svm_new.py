import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
 

def getDir(mode):
    return "./{0}.npy".format(mode)


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

features1 = load('unigram')
features2 = load('behav_feature')
features=features1

Y=np.load('./Data/label.npy')
Y=(Y==1)
all_fake = features[- cla]
all_real = features[cla]

m = min(len(all_fake), len(all_real))
m = 40000
features = np.concatenate((all_fake[:m], all_real[:m]), axis=0)
cla = np.array([False] * m + [True] * m)
assert(len(cla) == len(features))

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.33, random_state=42)


#classify
print('Start classify')
weight={}
weight[-1]=0.8
weight[1]=0.2
C = 1.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C,class_weight=weight).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C,class_weight=weight).fit(x_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C,class_weight=weight).fit(x_train, y_train)
lin_svc = svm.LinearSVC(C=C,class_weight=weight).fit(x_train, y_train)
'''
#Optimize SVM RBF
C_range = np.logspace(-2, 10, 10)
gamma_range = np.logspace(-9, 3, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(x_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
# c =2154, gamma=0.004641
'''
#evaluate
titles = ['LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
for i, clf in enumerate(( lin_svc, rbf_svc, poly_svc)):
	print(titles[i])
	pred=clf.predict(x_test)
	acc_test=np.sum(pred==y_test)/float(num_test)*100
	(p_test,r_test,f_test,s_test)=precision_recall_fscore_support(y_test,pred)
	pred=clf.predict(x_train)
	acc_train=np.sum(pred==y_train)/float(num_train)*100
	(p_train,r_train,f_train,s_train)=precision_recall_fscore_support(y_train,pred)
	print('---------------Training---------------')
	print('Accuracy:\t%f'%(acc_train))
	print('Precision:\t%f\t%f'%(p_train[0],p_train[1]))
	print('Recall:\t\t%f\t%f'%(r_train[0],r_train[1]))
	print('F1 score:\t%f\t%f'%(f_train[0],f_train[1]))
	print('---------------Testing---------------')
	print('Accuracy:\t%f'%acc_test)
	print('Precision:\t%f\t%f'%(p_test[0],p_test[1]))
	print('Recall:\t\t%f\t%f'%(r_test[0],r_test[1]))
	print('F1 score:\t%f\t%f\n'%(f_test[0],f_test[1]))
