
# coding: utf-8

# # 相関マップ
# ---
# 引数：raw_al45.csv/raw_al135.csvがあるディレクトリまでのパス
#
# ----
#
# 入力：raw_al45.csv/raw_al135.csv
#
# ----
#
# 出力：correlationmap.csv ボクセルごとに相関を算出したもの一覧
#
# ---
# ボクセルごとに相関を算出する．
# タスク1のZ-scoreには1を，タスク2のZ-scoreには-1をかけ，和を求める．値のものほど相関が大きいということになるため，昇順に並べ替えておく．

# In[223]:

print('############ Etc_correlationmap.py program excution ############')


# In[224]:

import numpy as np
import pandas as pd
import sys


# In[225]:

args = sys.argv
PATH = args[1]

# # jupyter notebookのときはここで指定
# PATH = '../SpmActive/20181119tm/RawData/'


# In[228]:

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


# In[229]:

# データフレームに格納されている値がstr型なのでfloat型にする
al45 = al45.astype(float)
al135 = al135.astype(float)


# In[230]:

# ボクセル数
voxNum = len(al45) // 4

# 何ボクセル目かをカウント
counter = 0

# ボクセル名取得用
voxNames = []

# ボクセルごとの相関格納用
cormap = []

for voxNo in range(voxNum):

    voxName = 'Voxel' + str(voxNo + 1)

    # ボクセルのデータを取得
    al45Vox = al45.loc[voxName]
    al135Vox = al135.loc[voxName]

    # array型に変換，各試行を一つにまとめる
    al45ar = np.array(al45Vox).reshape(-1)
    al135ar = np.array(al135Vox).reshape(-1)

    # 相関を求める
    cor_matrix = np.corrcoef(al45ar, al135ar)

    # 相関行列という形になっているので相関係数を取得
    cor = cor_matrix[0][1]

    # 求めた相関係数の絶対値を格納
    cormap = cormap + [abs(cor)]

    print(voxName + '( ' + str(counter+1) + ' / ' + str(voxNum) + ' ) : ' + str(cor))

    counter = counter + 1
    voxNames = voxNames + [voxName]


# In[231]:

# 相関一覧をデータフレーム化
cormap = pd.DataFrame(cormap)

# カラム名，インデックス名をつける
cormap.index = voxNames
cormap.columns = ['Correlation']


# In[232]:

# 相関の大き順に並べ替え
cormap_sort = cormap.sort_values('Correlation', ascending = False)


# In[233]:

# csv書き出し
PATH_cormap = PATH + 'correlationmap.csv'
cormap_sort.to_csv(PATH_cormap)


# In[221]:

# 相関一覧をデータフレーム化
cormap = pd.DataFrame(cormap)

# カラム名，インデックス名をつける
cormap.index = voxNames
cormap.columns = ['Correlation']
# 相関の大き順に並べ替え
cormap_sort = cormap.sort_values('Correlation', ascending = False)


# In[ ]:
