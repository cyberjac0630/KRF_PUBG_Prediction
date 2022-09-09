# -*- coding:utf-8 -*-
# Author:ZXW
# Time:2021/1/7
# 数据处理、分析
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import csv
import glob
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt

# sklearn模型
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as gbm
from sklearn.ensemble import GradientBoostingClassifier as gbdt
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error

# sklearn特征工程、数据准备和评估
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets.samples_generator import make_blobs
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn import clone
from matplotlib.figure import SubplotParams

# keras数据准备
from tensorflow.python.keras import models

# keras神经网络
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers

from scipy.interpolate import spline


date = np.load('data/train_demo.npy')
print(date.shape)

data = pd.DataFrame(date)
x = data.values[:, :24]
y = data.values[:, 24:25]
y = y.ravel()
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=32)  # 随机阈值是4
print('------')
from sklearn.model_selection import KFold

kf = KFold(5, False, 100)
print(X_train.shape)

class StackingAverageModels_build2():
    '''
    第一层的submodel是神经网络
    第二层的模型是其他模型。
    '''

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.doc_dir = None
        self.members = None
        self.n_models = None
        self.meta_model = None

    def load_all_models(self, n_models, doc_dir):
        all_models = list()
        for i in range(n_models):
            filename = os.path.join(doc_dir, 'model_' + np.str(i + 1) + '.h5')
            model = models.load_model(filename)
            all_models.append(model)
            print('>loaded %s' % (filename))

        self.members = all_models
        self.doc_dir = doc_dir
        self.n_models = n_models
        return all_models

    def stacked_dataset(self, inputX):
        '''
        第一层模型-建立模型并训练，输出预测结果
        '''
        stackX = None
        for model in self.members:
            # 做预测
            y_pred = model.predict(inputX, verbose=0)
            # 预测结果重塑成[row, members, probalities]
            if stackX is None:
                stackX = y_pred
            else:
                stackX = np.dstack((stackX, y_pred))
        # 将预测结果展开成,[rows, members * probalities]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))

        return stackX

    def fit_stacked_model(self, meta_model):
        '''
        # 创建训练集
        第二层模型-基于第一层预测结果，建立模型并训练
        return: 已训练好的模型
        '''
        inputX = self.X_test
        inputy = self.y_test

        stackedX = self.stacked_dataset(inputX)
        # 第二层的模型进行训练
        meta_model.fit(stackedX, inputy)
        self.meta_model = meta_model

        return meta_model

    def stacked_prediction_proba(self):
        '''
        基于第二层模型得到的预测结果
        '''
        # 创建训练数据集
        stackedX = self.stacked_dataset(self.X_test)
        # 做预测
        model = self.meta_model
        y_pred_roc = model.predict_proba(stackedX)[:,1]
        return y_pred_roc

    def stacked_prediction(self):
        '''
        基于第二层模型得到的预测结果
        '''
        # 创建训练数据集
        stackedX = self.stacked_dataset(self.X_test)
        # 做预测
        model = self.meta_model
        y_pred = model.predict(stackedX)
        return y_pred


def DNN_base_v1(X_train, y_train):
    model = models.Sequential()
    model.add(
        layers.Dense(96, activation='elu', kernel_regularizer=regularizers.l2(0.005), input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=2000, batch_size=300, validation_split=0.2, verbose=0, shuffle=True)
    results_train = model.evaluate(X_train, y_train)

    print('accuracy: %s' % (results_train))
    return model


def DNN_fit_and_save(X_train, y_train, doc_dir):
    if os.path.exists(doc_dir) == True:
        pass
    else:
        os.makedirs(doc_dir)
    for i, (X_train_index, y_train_index) in enumerate(kf.split(X_train)):
        print(i)
        x = X_train[X_train_index]
        y = y_train[X_train_index]
        model = DNN_base_v1(x, y)
        filename = os.path.join(doc_dir, 'model_' + np.str(i + 1) + '.h5')
        model.save(filename)
        print('>save %s' % (filename))


doc_dir = r'./tmp_models'
print(X_train.shape, y_train.shape)
# dnn = DNN_fit_and_save(X_train, y_train, doc_dir)

lr = LR(random_state=123, verbose=0, solver='liblinear')
svm_clf2 = SVC(kernel='rbf', class_weight='balanced', random_state=123)
dt = DT(max_depth=4, random_state=123)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
rdf = RandomForestClassifier(random_state=123)
gbm_sklearn_model = gbdt(random_state=123)
xgb_model = xgb.XGBClassifier(seed=123)
gbm_model = gbm.LGBMClassifier(random_state=123)


def caculate_main_p(pre_labels, target):
    pre_labels_reg = np.array(pre_labels)
    target_labels_reg = np.array(target)
    # print(pre_labels_reg)
    # print(target_labels_reg)
    Recall = recall_score(target_labels_reg, pre_labels_reg, average='micro')
    Precision = precision_score(target_labels_reg, pre_labels_reg,average='micro')
    F1 = f1_score(target_labels_reg, pre_labels_reg,average='micro')
    Recall = round(Recall, 4)
    Precision = round(Precision, 4)
    F1 = round(F1, 4)
    print(Precision, Recall, F1)

    return Precision, Recall, F1




plt.figure()
lw = 1

#1

aa = StackingAverageModels_build2(X_train, y_train, X_test, y_test)
aa.load_all_models(doc_dir=r'./tmp_models', n_models=5)
aa.fit_stacked_model(meta_model=dt)
y_pred = aa.stacked_prediction()
y_pred_roc = aa.stacked_prediction_proba()

scores = np.array(y_pred_roc)
roc_y = np.array(y_test)

# f = open('pred/DT.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(y_pred)

# f = open('pred/true.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(y_test)

# 2021/1/9
# fpr, tpr, thresholds = metrics.roc_curve(roc_y, scores)
# auc = metrics.auc(fpr, tpr)
Precision = caculate_main_p(y_pred, y_test)

# plt.plot(fpr, tpr, color='firebrick',
#          lw=lw, label='DT (AUC = %0.2f)' % auc)

#2
bb = StackingAverageModels_build2(X_train, y_train, X_test, y_test)
bb.load_all_models(doc_dir=r'./tmp_models', n_models=5)
bb.fit_stacked_model(meta_model=nb)
y_pred = bb.stacked_prediction()
y_pred_roc = bb.stacked_prediction_proba()

scores = np.array(y_pred_roc)

# f = open('pred/NB.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(y_pred)

# fpr, tpr, thresholds = metrics.roc_curve(roc_y, scores)
# auc = metrics.auc(fpr, tpr)
Precision = caculate_main_p(y_pred, y_test)

# plt.plot(fpr, tpr, color='gray',
#          lw=lw, label='NB (AUC = %0.2f)' % auc)

#3
cc = StackingAverageModels_build2(X_train, y_train, X_test, y_test)
cc.load_all_models(doc_dir=r'./tmp_models', n_models=5)
cc.fit_stacked_model(meta_model=gbm_model)
y_pred = cc.stacked_prediction()
y_pred_roc = cc.stacked_prediction_proba()

scores = np.array(y_pred_roc)

# f = open('pred/GBM.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(y_pred)

# fpr, tpr, thresholds = metrics.roc_curve(roc_y, scores)
# auc = metrics.auc(fpr, tpr)
Precision = caculate_main_p(y_pred, y_test)

# plt.plot(fpr, tpr, color='gold',
#          lw=lw, label='GBM (AUC = %0.2f)' % auc)

#4
dd = StackingAverageModels_build2(X_train, y_train, X_test, y_test)
dd.load_all_models(doc_dir=r'./tmp_models', n_models=5)
dd.fit_stacked_model(meta_model=xgb_model)
y_pred = dd.stacked_prediction()
y_pred_roc = dd.stacked_prediction_proba()

scores = np.array(y_pred_roc)

# f = open('pred/XGB.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(y_pred)

# fpr, tpr, thresholds = metrics.roc_curve(roc_y, scores)
# auc = metrics.auc(fpr, tpr)
Precision = caculate_main_p(y_pred, y_test)

# plt.plot(fpr, tpr, color='darkblue',
#          lw=lw, label='XGB (AUC = %0.2f)' % auc)

#5
ee = StackingAverageModels_build2(X_train, y_train, X_test, y_test)
ee.load_all_models(doc_dir=r'./tmp_models', n_models=5)
ee.fit_stacked_model(meta_model = rdf)
y_pred = ee.stacked_prediction()
y_pred_roc = ee.stacked_prediction_proba()

scores = np.array(y_pred_roc)

# f = open('pred/RDF.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(y_pred)

# fpr, tpr, thresholds = metrics.roc_curve(roc_y, scores)
# auc = metrics.auc(fpr, tpr)
Precision = caculate_main_p(y_pred, y_test)

# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='RDF (AUC = %0.2f)' % auc)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
