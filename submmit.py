# -*- coding:utf-8 -*-
# Author:ZXW
# Time:2021/1/11
import pandas as pd
import numpy as np
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tensorflow.python.keras import models
from keras import models

date_train = np.load('data/train_demo_train_300000.npy')

out_data = []
with open('data/test_V2.csv', 'r', encoding='utf-8') as f:
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

test = pd.read_csv('data/test_V2.csv')
test1 = test.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]]

date_train = pd.DataFrame(date_train)
date_test = pd.DataFrame(date_test)
x_train = date_train.values[:, :24]
y_train = date_train.values[:, 24:25]
x_test = date_test.values[:, :24]
y = y_train.ravel()

kf = KFold(5, False, 100)


class StackingAverageModels_build2():
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
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

    def stacked_dataset(self, inputx):
        stackx = None
        for model in self.members:
            # print(inputx)
            y_pred = model.predict(inputx, verbose=0)
            if stackx is None:
                stackx = y_pred
            else:
                stackx = np.dstack((stackx, y_pred))
        stackx = stackx.reshape((stackx.shape[0], stackx.shape[1] * stackx.shape[2]))
        return stackx

    def fit_stacked_model(self, meta_model):
        inputx = self.x_train
        inputy = self.y_train
        stackedx = self.stacked_dataset(inputx)
        meta_model.fit(stackedx, inputy)
        self.meta_model = meta_model
        return meta_model

    def stacked_prediction(self):
        stackedx = self.stacked_dataset(self.x_test)
        model = self.meta_model
        y_pred = model.predict(stackedx)
        return y_pred


doc_dir = r'./tmp_models'

rdf = RandomForestClassifier(random_state=123)

aa = StackingAverageModels_build2(x_train, y_train, x_test)
aa.load_all_models(doc_dir=r'./tmp_models', n_models=5)
aa.fit_stacked_model(meta_model = rdf)
y_pred = aa.stacked_prediction()

pre = pd.DataFrame(y_pred)
id = test.iloc[:,0:1]
df = pd.concat([id,pre], axis=1)
df.rename(index=str,columns={ 0:"winPlacePerc"}, inplace = True)
df.to_csv('submission.csv', index=False)