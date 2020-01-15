import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#data preprocessing
def conv2zero(dataset):
	for i in range(len(dataset)):
		for j in range(len(dataset.iloc[i,:])):
			if dataset.iloc[i, j] == '?':
				dataset.iloc[i, j] = 0
			if '0x' in str(dataset.iloc[i, j]):
				int(str(dataset.iloc[i, j]), base=16)
				#dataset.iloc[i, j] =16#


def label_encode(x, cols):
	for i in cols:
		labelencoder = LabelEncoder()
		x.iloc[:, i] = str(x.iloc[:, i])
		x.iloc[:, i] = labelencoder.fit_transform(x.iloc[:, i])

def one_hot_encode(x, cols):
	for i in cols:
		onehotencoder = OneHotEncoder(categorical_features = [i])
		x = onehotencoder.fit_transform(x).toarray()

def model_training(model, x_train, x_test, y_train, y_test):
	print("***************************************************************")
	print(model.clf)
	print("model training...")
	t0 = time.time();
	model.fit(x_train, y_train)#在训练集训练模型
	train_time = time.time() - t0

	t0 = time.time();
	expected = y_test
	predicted = model.predict(x_test)#在测试集进行测试
	test_time = time.time() - t0

	accuracy = accuracy_score(expected, predicted)
	recall = recall_score(expected, predicted, average="binary")
	precision = precision_score(expected, predicted , average="binary")
	f1 = f1_score(expected, predicted , average="binary")
	cm = confusion_matrix(expected, predicted)
	tpr = float(cm[0][0])/np.sum(cm[0])
	fpr = float(cm[1][1])/np.sum(cm[1])

	print(cm)
	print("tpr:%.3f" %tpr)
	print("fpr:%.3f" %fpr)
	print("accuracy:%.3f" %accuracy)
	print("precision:%.3f" %precision)
	print("recall:%.3f" %recall)
	print("f-score:%.3f" %f1)
	print("train_time:%.3fs" %train_time)
	print("test_time:%.3fs" %test_time)
	print("***************************************************************")

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

print("load data...")
train = pd.read_csv('train_sample_3000.csv',low_memory=False, header=None)
train, _ = train_test_split(train, test_size=0.9)
train, _ = train_test_split(train, test_size=0.9)
train, _ = train_test_split(train, test_size=0.9)


    # col_names = []
    #train.to_csv("train.csv")


#test = pd.read_csv('test_sample_3000.csv',low_memory=False, header=None)


print("encoding categorical data...")
conv2zero(train)
#conv2zero(test)

cols = [75, 76, 77, 78, 79, 118]
train.drop(columns=['75','76','77','78','79','118'], inplace=True,axis=1)

train, test = train_test_split(train, test_size=0.5)

label_encode(train, cols)
one_hot_encode(train, cols)

label_encode(test, cols)
one_hot_encode(test, cols)

x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 154].values

x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, 154].values

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


'''
model_gbdt = GradientBoostingClassifier(n_estimators=100)
model_lr = LogisticRegression()
model_adaboost = AdaBoostClassifier()
model_rf = RandomForestClassifier()
model_training(model_gbdt, x_train, x_test, y_train, y_test)
model_training(model_lr, x_train, x_test, y_train, y_test)
model_training(model_adaboost, x_train, x_test, y_train, y_test)
model_training(model_rf, x_train, x_test, y_train, y_test)
'''
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

#model training
SEED = 0
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

#model_training(rf, x_train, x_test, y_train, y_test)
#model_training(et, x_train, x_test, y_train, y_test)
#model_training(ada, x_train, x_test, y_train, y_test)
#model_training(gb, x_train, x_test, y_train, y_test)
model_training(svc, x_train, x_test, y_train, y_test)