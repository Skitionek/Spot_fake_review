import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
#setting
np.set_printoptions(threshold=np.nan)
# file name
data_fol='./ML/behavioral_feature_extraction/custom_data/'
rate_file='rating_deviation.npy'
user_file='user_prod_inburst.npy'
bbr_file='BBR.npy'

#load
rate=np.load(data_fol+rate_file)
user=np.load(data_fol+user_file)
bbr=np.load(data_fol+bbr_file)
label=np.load('./Data/label.npy')
meta = np.loadtxt('./Data/YelpZip/metadata', dtype=np.dtype("i4, i4, f4, i4, S10"))

#create dict
rate_dict={}
user_dict={}
bbr_dict={}
for i in range(rate.shape[0]):
	rate_dict[rate[i,0]]=rate[i,1]
	bbr_dict[bbr[i,0]]=bbr[i,1]

for i in range(user.shape[0]):
	if user[i,0] in user_dict:
		user_dict[user[i,0]][user[i,1]]=user[i,2]
	else:
		user_dict[user[i,0]]={}
		user_dict[user[i,0]][user[i,1]]=user[i,2]
rate=[]
user=[]
bbr=[]

# create input

X=np.zeros([meta.shape[0],3])
Y=label
for i in range(meta.shape[0]):
	X[i,0]=rate_dict[meta[i][0]]
	X[i,1]=bbr_dict[meta[i][0]]
	X[i,2]=user_dict[meta[i][0]][meta[i][1]]
meta=[]

#train and test set
num_data=10000 #X.shape[0]
ratio=0.8
num_train=int(np.ceil(num_data*ratio))
num_test=num_data-num_train
x_train=X[0:num_train,:]
x_test=X[num_train:num_data,:]
y_train=Y[0:num_train]
y_test=Y[num_train:num_data]
X=[]
Y=[]

# normalize
scaler = preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)  
x_test=scaler.transform(x_test)

#classify
weight={}
weight[-1]=0.8
weight[1]=0.2
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C,class_weight=weight).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C,class_weight=weight).fit(x_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C,class_weight=weight).fit(x_train, y_train)
lin_svc = svm.LinearSVC(C=C,class_weight=weight).fit(x_train, y_train)

#evaluate
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
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
	print('Recall\t\t%f\t%f'%(r_train[0],r_train[1]))
	print('F1 score:\t%f\t%f'%(f_train[0],f_train[1]))
	print('---------------Testing---------------')
	print('Accuracy:\t%f'%acc_test)
	print('Precision:\t%f\t%f'%(p_test[0],p_test[1]))
	print('Recall\t\t%f\t%f'%(r_test[0],r_test[1]))
	print('F1 score:\t%f\t%f\n'%(f_test[0],f_test[1]))
