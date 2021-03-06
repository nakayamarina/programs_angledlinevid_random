
# coding: utf-8

# # SVMによるボクセルごとの学習と性能評価（時系列解析）
# ----
#
# 引数：raw_al45.csv/raw_al135.csvがあるディレクトリまでのパス
#
# ----
#
# 入力：raw_al45.csv/raw_al135.csv
#
# ----
#
# 出力：ACCURACY[loo or k_cv]_VOXtimeseries[1時系列あたりのスキャン数]_SVM.csv ボクセルごとの識別性能評価結果一覧
#
# ----
# ボクセルごとにあるデータにおけるNスキャン分の時系列データをテストデータ，テストデータを除いた残りのデータにおけるNスキャン分の時系列データ1ずつずらしながらを学習データとして取得し，SVMを用いて学習，交差検証法を用いて識別性能評価を行う．
# ベクトル：各ボクセルにおけるデータにおけるNスキャン分の時系列データ

# In[1]:

print('############ ML_SVM_VOXtimeseries.py program excution ############')


# In[2]:

import numpy as np
import pandas as pd
import sys

from sklearn import svm


# In[356]:

args = sys.argv
PATH = args[1]

# # jupyter notebookのときはここで指定
# PATH = '../Active/20181119mt/RawData/'

# 1時系列あたりのスキャン数
N = 30

# 検証手法
col_name = 'leave-one-out'

# 試行数
runNum = 4


# ## TsShift関数
# 引数としてあるボクセルにおける全試行分のデータをdata，タスクを見分けるための番号をlabelに受け取る．
# 各試行で1ずつずらしながらNスキャン分の時系列データを取得する．全試行の時系列データをまとめて返す．

# In[357]:

def TsShift(data, label):

    # 1ボクセルあたりで取得する時系列データ格納用データフレーム
    ts_all = pd.DataFrame(index = [], columns = [])

    for runNo in range(runNum):

        # 1試行あたりので得られる全データ
        runData = data.iloc[runNo, :]

        # 時系列として扱う区間の始まり
        ts_fst = 0

        # 時系列として扱う区間の終わり
        ts_end = N

        # 1ずつずらしながら時系列データを取得，結合
        while ts_end <= len(runData):

            # 時系列として扱う区間の始まり，区間の終わり，その区間（Nスキャン分）の時系列データの順でリスト化
            ts = [label] + [runNo + 1] + [ts_fst + 1] + [ts_end] + list(runData[ts_fst:ts_end])

            # データフレーム化，結合
            ts = pd.DataFrame(ts).T

            ts_all = pd.concat([ts_all, ts])

            # 1ずつずらす
            ts_fst = ts_fst + 1
            ts_end = ts_end + 1


    return ts_all


# ## SVM_LOO関数
# 学習と評価に用いるデータをdataで受け取り．
# データからテストデータ，テストデータラベル，教師データ，教師データラベルを生成．
# テストデータはdataにおける全時系列データ，教師データは条件に当てはまる時系列データであり，
# テストデータに用いるデータごとに識別できたかできなかったか（1か0）を取得，全テストデータで識別できた(1)の割合を算出（leave-one-outと同じ要領）．
# 得られた割合をパーセント表記にし，main関数へ返す．

# In[3]:

def SVM_LOO(data):

    scores = np.zeros(len(data))

    for i in range(len(data)):

        print("--------" + str(i + 1) + " / " + str(len(data)) + "---------")

        # テストデータの情報（タスク，試行数，時系列データの始まり，終わり）を取得
        test_label = data.iloc[i, 0]
        test_run = data.iloc[i, 1]
        test_fst = data.iloc[i, 2]
        test_end = data.iloc[i, 3]


        ###### テストデータ

        # 時系列データのみを抽出
        X_test = data.iloc[i, 4:len(data.columns)]

        # ベクトル化
        X_test = X_test.as_matrix()
        X_test = X_test.reshape(1, -1)

        # ラベルを作成
        y_test = np.array([test_label])
        y_test = y_test.reshape(1, -1)

        ###### 教師データ

        # テストデータではない，テストデータに含まれるZ-scoreを含まないものを取得
        traindata = data[(data['label'] != test_label) | (data['run'] != test_run) | (data['fst'] > test_end) | (data['end'] < test_fst)]

        # 時系列データのみを抽出
        X_train = traindata.iloc[:, 4:len(data.columns)]

        # ベクトル化
        X_train = X_train.as_matrix()

        # ラベルを作成
        y_train = np.array(list(traindata['label']))


        print("Test Data : " + str(test_fst) + "-" + str(test_end) + " / Train Data Num : " + str(len(traindata)))


        # 線形SVMのインスタンスを生成
        model = svm.SVC(kernel = 'linear', C=1)

        # モデルの学習
        model.fit(X_train, y_train)

        # 評価結果を格納
        scores[i] = model.score(X_test, y_test)

    # 評価結果の平均を求める
    result = scores.mean()


    # パーセント表記へ
    result = round(result * 100, 1)

    print(str(scores) + '\n')

    return result


# In[4]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_al45 = PATH + 'raw_al45.csv'
    PATH_al135 = PATH + 'raw_al135.csv'

    # csvファイル読み込み
    # headerは設定せず，転置後にset_index()する（header = 0にすると列名が変えられる）
    al45 = pd.read_csv(PATH_al45, header = None, index_col = 0).T
    al45.columns = range(0, len(al45.columns))
    al45 = al45.set_index(0)

    al135 = pd.read_csv(PATH_al135, header = None, index_col = 0).T
    al135.columns = range(0, len(al135.columns))
    al135 = al135.set_index(0)


    # In[360]:

    # ボクセル数
    voxNum = len(al45) // 4

    # 全ボクセルの識別率を格納するデータフレーム
    voxAc = pd.DataFrame(index = range(voxNum), columns = [col_name])

    counter = 0
    csvcounter = 0
    voxNames = []


    for voxNo in range(voxNum):

        voxName = 'Voxel' + str(voxNo + 1)

        print(voxName + '( ' + str(counter) + ' / ' + str(voxNum) + ' )')

        # ボクセルのデータを取得
        al45Vox = al45.loc[voxName]
        al135Vox = al135.loc[voxName]

        # ボクセルにおける時系列データを取得
        al45VoxTs = TsShift(al45Vox, 0)
        al135VoxTs = TsShift(al135Vox, 1)

        # 全タスクを縦結合
        VoxTs = pd.concat([al45VoxTs, al135VoxTs])

        # 0-3列目は条件判定用の要素，要素名をつけておく
        col_names = list(VoxTs.columns)
        col_names[0:4] = ['label', 'run', 'fst', 'end']
        VoxTs.columns = col_names

        VoxTs.index = range(0,len(VoxTs))

        # 学習と評価
        result_vox = SVM_LOO(VoxTs)

        print(result_vox)

        # データフレームに格納
        voxAc.at[voxNo, :] = result_vox

        # 途中経過見る用
        # 何ボクセルで一度出力するか
        midNum = 1000

        if (counter % midNum == 0) and (counter != 0):

            PATH_test = PATH + 'ACMID' + str(csvcounter) + '[loo]_VOXtimeseries' + str(N) +'_SVM.csv'
            print(PATH_test)
            MidVoxAc = voxAc.iloc[(csvcounter * midNum):((csvcounter + 1) * midNum), :]
            MidVoxAc.index = voxNames[(csvcounter * midNum):((csvcounter + 1) * midNum)]
            MidVoxAc.to_csv(PATH_test, index = True)

            csvcounter = csvcounter + 1

        counter = counter + 1
        voxNames = voxNames + [voxName]




    # In[352]:

    # csv書き出し
    PATH_RESULT = PATH + 'ACCURACY[loo]_VOXtimeseries' + str(N) +'_SVM.csv'
    voxAc.to_csv(PATH_RESULT, index = True)

    # 行名つける
    voxAc.index = voxNames
    # csv書き出し
    PATH_RESULT = PATH + 'ACCURACY[loo]_VOXtimeseries' + str(N) +'_SVM.csv'
    voxAc.to_csv(PATH_RESULT, index = True)

    # In[ ]:


    # In[ ]:
