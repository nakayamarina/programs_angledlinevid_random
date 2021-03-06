
# coding: utf-8

# # 一次元CNNによる学習と識別性能評価（時系列解析）

# ---
#
# 引数：raw_al45.csv/raw_al135.csvがあるディレクトリまでのパス
#
# ---
#
# 入力：raw_al45.csv/raw_al135.csv
#
# ---
#
# 出力：ACCURACY[loo or k_cv]_timeseries_1dCNN.csv　識別性能評価結果一覧
#
# ---
#
# 1dCNNを用いて学習し，交差検証法（k-分割交差検証，leave-one-out交差検証）を用いて識別性能評価を行う．
# ベクトル：各ボクセルの時系列データ

# In[1]:

print('########### ML_1dCNN_timeseries.py program excution ############')


# In[2]:

import pandas as pd
import sys

# matplotlib inline部分は.pyの時にはコメントアウトしないとエラー出る！

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.utils import np_utils

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# コマンドライン引数でraw_al45.csv/raw_al135.csvファイルがあるディレクトリまでのパスを取得

# In[3]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../Active/20181119mt/RawData/'

# 検証手法
col_name = 'leave-one-out'


# ## ONEdCNN_LOO関数

# 引数として教師データをX，ラベルをyで受け取る．
# 交差検証法の一つleave-one-out交差検証で識別精度評価を行う．
#
# * (1個をテストデータ，残りを教師データにして学習・評価) * すべてのデータ個
# * 得られたすべてのデータ個の評価結果（識別率）の平均を求めてパーセントに直す
# * 評価結果（識別率）をmain関数に返す

# In[4]:

def ONEdCNN_LOO(X, y):

    # 識別率を格納する配列
    LOOscore = np.zeros(len(X))

    # ベクトルの長さを格納しておく
    bach_size = X.shape[1]


    # 1個をテストデータ，残りを教師データにして学習・評価
    # すべてのデータに対して行う
    for i in range(len(X)):

        print('------ ' + str(i) + ' / ' + str(len(X)) + '回 -----')

        # テストデータ
        X_test = X[i]
        y_test = y[i]

        # テストデータとして使用するデータを除いた教師データを作成
        X_train = np.delete(X, i, 0)
        y_train = np.delete(y, i, 0)


        # （データ数, ベクトルの長さ，1）という形にリシェイプする
        X_train = np.reshape(X_train, (-1, bach_size, 1))
        X_test = np.reshape(X_test, (-1, bach_size, 1))

        # ダミー変数に変換：何分類するかによって数字を書き換える
        y_train = np_utils.to_categorical(y_train, 2)
        y_test = np_utils.to_categorical(y_test, 2)


        # 1次元CNNのインスタンスを作成
        # 参考文献（Time series classification via TDA）に従ってパラメータを設定
        # 1st Conv1d : kernel_number = 7, kernel_size = 6
        # 1st MacPooling : kernel_number = 7
        # 2nd Conv1d : kernel_number = 7, kernel_size = 2
        # 2nd MacPooling : kernel_number = 3
        # Flattenで1次元に
        model = Sequential()
        model.add(Conv1D(7, 6, padding='same', input_shape=(bach_size, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(7, padding='same'))
        model.add(Conv1D(7, 2, padding='same', activation='relu'))
        model.add(MaxPooling1D(3, padding='same'))

        model.add(Flatten())

        # 何分類するかによってunitsを書き換える
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

        # 学習
        history = model.fit(X_train, y_train, epochs=100)

        # 評価結果（識別率）を格納
        score = model.evaluate(X_test, y_test)
        LOOscore[i] = score[1]



    # 評価結果（識別率）の平均を求める
    result = LOOscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print(str(LOOscore) + '\n')

    return result


# ## ONEdCNN_kCV関数

# 引数として教師データをX，ラベルをyで受け取る．
# 交差検証法の一つk-分割交差検証で識別精度評価を行う．
#
# * 学習
# * (k分割し，1グループをテストデータ，残りグループを教師データにして評価) * k
# * 得られたk個の評価結果（識別率）の平均を求めてパーセントに直す
# * 評価結果（識別率）をmain関数に返す

# In[5]:

def ONEdCNN_kCV(X, y):


    # 識別率を格納する配列
    kCVscore = np.zeros(cv_k)

    # ベクトルの長さを格納しておく
    bach_size = X.shape[1]

    # 分割数
    kf = KFold(n_splits=cv_k, shuffle=False)
    # 繰り返し回数
    i = 0

    for train_index, eval_index in kf.split(X):

        print('------ ' + str(i) + ' / ' + str(cv_k) + '回 -----')

        X_train, X_test = X[train_index], X[eval_index]
        y_train, y_test = y[train_index], y[eval_index]

        # （データ数, ベクトルの長さ，1）という形にリシェイプする
        X_train = np.reshape(X_train, (-1, bach_size, 1))
        X_test = np.reshape(X_test, (-1, bach_size, 1))

        # ダミー変数に変換：何分類するかによって数字を書き換える
        y_train = np_utils.to_categorical(y_train, 2)
        y_test = np_utils.to_categorical(y_test, 2)


        # 1次元CNNのインスタンスを作成
        # 参考文献（Time series classification via TDA）に従ってパラメータを設定
        # 1st Conv1d : kernel_number = 7, kernel_size = 6
        # 1st MacPooling : kernel_number = 7
        # 2nd Conv1d : kernel_number = 7, kernel_size = 2
        # 2nd MacPooling : kernel_number = 3
        # Flattenで1次元に
        model = Sequential()
        model.add(Conv1D(7, 6, padding='same', input_shape=(bach_size, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(7, padding='same'))
        model.add(Conv1D(7, 2, padding='same', activation='relu'))
        model.add(MaxPooling1D(3, padding='same'))

        model.add(Flatten())

        # 何分類するかによってunitsを書き換える
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

        # 学習
        history = model.fit(X_train, y_train, epochs=100)

        # 評価結果（識別率）を格納
        score = model.evaluate(X_test, y_test)
        kCVscore[i] = score[1]

        i = i + 1


    # 評価結果（識別率）の平均を求める
    result = kCVscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print('k = ' + str(cv_k) + '：' + str(kCVscore))

    return result


# ## TrainingData関数
# 引数として読み込みたいタスクごとのデータをdata1/data2で受け取る．
# * 機械学習にかけれるようにデータのベクトル化とラベルを作成
# * ベクトル化したデータとラベルをmain関数に返す

# In[6]:

def TrainingData(data1, data2):

    # 各タスクのデータを縦結合
    all_data = pd.concat([data1, data2], axis = 0)

    # ベクトル化
    X = all_data.as_matrix()

    # ラベル作成 data1 = 0, data2 = 1
    label_data1 = np.zeros(len(data1.index))
    label_data2 = np.ones(len(data2.index))

    y = np.r_[label_data1, label_data2]


    return X, y


# ## main関数

# In[7]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_al45 = PATH + 'raw_al45.csv'
    PATH_al135 = PATH + 'raw_al135.csv'

    # csvファイル読み込み，時系列解析のため行列を入れ替える
    al45 = pd.read_csv(PATH_al45, header = 0, index_col = 0)
    al45 = al45.T

    al135 = pd.read_csv(PATH_al135, header = 0, index_col = 0)
    al135 = al135.T

    # データとラベルの準備
    data, labels = TrainingData(al45, al135)


# In[8]:

# 学習とleave-one-out交差検証法

print('leave-one-out Cross-Validation')

result_loo = ONEdCNN_LOO(data, labels)

# データフレーム化
result_loo = pd.DataFrame({col_name:[result_loo]}, index = ['TimeSeries + 1dCNN'])
print(result_loo)


# In[9]:

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[loo]_timeseries_1dCNN.csv'
result_loo.to_csv(PATH_RESULT, index = True)


# In[ ]:




# In[ ]:
