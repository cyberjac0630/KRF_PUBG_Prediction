# -*- coding:utf-8 -*-
# Author:ZXW
# Time:2021/1/7
# 数据处理、分析
import pandas as pd
import numpy as np
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tensorflow.python.keras import models
from keras import models

date_train = np.load('../input/model_trained/train_demo_train_300000.npy')

out_data = []
with open('../input/pubg-finish-placement-prediction/test_V2.csv', 'r', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    header_status = 0
    for row in f_csv:
        if header_status == 0:
            header_status = header_status + 1
            continue
        del row[1]
        del row[1]
        del row[13]
        out_data.append(row[1:25])
    f.close()
date_test = np.array(out_data)

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
test1 = test.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]]

date_train = pd.DataFrame(date_train)
date_test = pd.DataFrame(date_test)
x_train = date_train.values[:, :24]
y_train = date_train.values[:, 24:25]
x_test = date_test.values[:, :24]
y = y_train.ravel()

kf = KFold(5, False, 100)


class StackingAverageModels_build2():
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
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
        stackX = None
        for model in self.members:
            y_pred = model.predict(inputX, verbose=0)
            if stackX is None:
                stackX = y_pred
            else:
                stackX = np.dstack((stackX, y_pred))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
        return stackX

    def fit_stacked_model(self, meta_model):
        inputX = self.X_train
        inputy = self.y_train
        stackedX = self.stacked_dataset(inputX)
        meta_model.fit(stackedX, inputy)
        self.meta_model = meta_model
        return meta_model

    def stacked_prediction(self):
        stackedX = self.stacked_dataset(self.X_test)
        model = self.meta_model
        y_pred = model.predict(stackedX)
        return y_pred


doc_dir = r'./model_trained'

rdf = RandomForestClassifier(random_state=123)

aa = StackingAverageModels_build2(X_train, y_train, X_test)
aa.load_all_models(doc_dir=r'../input/model_trained', n_models=5)
aa.fit_stacked_model(meta_model = rdf)
y_pred = aa.stacked_prediction()

pre = pd.DataFrame(y_pred)
id = test.iloc[:,0:1]
df = pd.concat([id,pre], axis=1)
df.rename(index=str,columns={ 0:"winPlacePerc"}, inplace = True)
df.to_csv('submission.csv', index=False)


