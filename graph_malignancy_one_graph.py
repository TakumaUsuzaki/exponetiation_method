import numpy as np
import random
import glob
import os
import shutil
import cv2
import re
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as pp
from scipy import stats
from statistics import mean, median, variance, stdev
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import pprint

acc_list_all = []

mal = 2

input_val = "./Malignancy1vs{0}".format(mal)
lst = [x for x in np.arange(2, 21.0, 1.0)]

accuracy = []
TP = []
FP = []
FN = []
TN = []
EP = []

# 各指数について
for i in lst:
    acc_lst = []
    tp_lst = []
    fp_lst = []
    fn_lst = []
    tn_lst = []
    epoch_max = []

    a = []
    b = []
    c = []
    d = []

    path_one = glob.glob(input_val + "/exp{0}_n?.json".format(int(i)))
    path_ten = glob.glob(input_val + "/exp{0}_n??.json".format(int(i)))
    path_list = path_one + path_ten

    # 各jsonについて
    for j in path_list:
        data_json = open(j, 'r')
        data = json.load(data_json)

        # test_accについて
        li_acc = [x for x in data['test_acc']]  # 40epochs分のリスト
        max_acc = max(li_acc)  # 40epochsのうち最大となるacc
        acc_lst.append(max_acc)
        max_index = li_acc.index(max_acc)  # その時のindex

        epoch_max.append(max_index)

        # とりあえず各jsonでの最大となるacc(要はmax_acc)の時のindex(max_index)を取得する。
        # TPについて
        li_tp = [x for x in data['TP']]
        tp = li_tp[max_index]
        tp_lst.append(tp)

        # FPについて
        li_fp = [x for x in data['FP']]
        fp = li_fp[max_index]
        fp_lst.append(fp)

        # FNについて
        li_fn = [x for x in data['FN']]
        fn = li_fn[max_index]
        fn_lst.append(fn)

        # TNについて
        li_tn = [x for x in data['TN']]
        tn = li_tn[max_index]
        tn_lst.append(tn)

    acc_ave = np.mean(acc_lst)
    m = np.max(acc_lst)
    index_num = acc_lst.index(m)

    accuracy.append(acc_lst)

    epoch_mean = np.mean(epoch_max)
    EP.append(epoch_mean)
    TP.append(tp_lst)
    FP.append(fp_lst)
    FN.append(fn_lst)
    TN.append(tn_lst)

TP = np.array(TP)
# print("TP:", TP)
FP = np.array(FP)
FN = np.array(FN)
TN = np.array(TN)
"""
感度，特異度，陽性的中率，陰性的中率
"""
se = [TP[i] / (TP[i] + FN[i]) for i in range(len(TP))]
sp = [TN[i] / (TN[i] + FP[i]) for i in range(len(TN))]
ppv = [TP[i] / (TP[i] + FP[i]) for i in range(len(TP))]
npv = [TN[i] / (TN[i] + FN[i]) for i in range(len(TN))]

type_se = ['Mean sensitivity' for i in range(950)]
type_sp = ['Mean specificity' for i in range(950)]
type_ppv = ['Positive predicting value' for i in range(950)]
type_npv = ['Negative predicting value' for i in range(950)]
type_acc = ['Mean accuracy' for i in range(950)]

all_x_axis = []
for n in [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]:
    x_axis = [n for i in range(50)]
    all_x_axis.extend(x_axis)

all_lst_se = []
for n in se:
    all_lst_se.extend(n)

all_lst_sp = []
for n in sp:
    all_lst_sp.extend(n)

all_lst_ppv = []
for n in ppv:
    all_lst_ppv.extend(n)

all_lst_npv = []
for n in npv:
    all_lst_npv.extend(n)

acc_list_all.append(accuracy)
accuracy_lst = acc_list_all[0]
accuracy2vs3 = []
for n in accuracy_lst:
    accuracy2vs3.extend(n)

print(len(all_x_axis))
print(len(all_lst_se))
print(len(type_se))
df1 = pd.DataFrame(
    {'Exponent': all_x_axis, 'Value': all_lst_se, 'Statistics': type_se})
df2 = pd.DataFrame(
    {'Exponent': all_x_axis, 'Value': all_lst_sp, 'Statistics': type_sp})
# df3 = pd.DataFrame(
#     {'Exponent': all_x_axis, 'Value': all_lst_ppv, 'Statistics': type_ppv})
# df4 = pd.DataFrame(
#     {'Exponent': all_x_axis, 'Value': all_lst_npv, 'Statistics': type_npv})
df5 = pd.DataFrame(
    {'Exponent': all_x_axis, 'Value': accuracy2vs3, 'Statistics': type_acc})

""" 4つのdfを統合する """
#df = pd.concat([df1, df2, df5])
df = pd.concat([df5, df1, df2])
print(df)

statistics_lst = [
    'Sensitivity',
    'Specificity',
    'Accuracy',
]

fig = plt.figure(figsize=(20, 15))
plt.subplots_adjust(wspace=0.2, hspace=0.4)
sns.set_style("whitegrid", {'grid.linestyle': '--'})

""" linelpot 本体 """
# palette=['#2ecc71','#3498db','#9b59b6']
ax = sns.lineplot(
    x='Exponent',
    y='Value',
    data=df,
    palette=['black','black','black'],
    hue='Statistics',
    style='Statistics',
    markers=True,
    linewidth=2,
    markersize=17,
    ci=False)

""" legend 削除 """
ax.legend(handlelength=9, markerscale=2.2, loc = 'best')
plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title

""" x軸とy軸の設定 """
ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
ax.set_ylim([0.72, 0.84])
plt.xlabel("Exponent", fontsize=25, labelpad=10)
plt.ylabel("Value", fontsize=25, labelpad=10)

""" titleの設定 """
plt.title("Malignancy1vs{0}".format(mal), fontsize=30, pad=20)
plt.tick_params(axis="x",labelsize=30)
plt.tick_params(axis="y",labelsize=30)

plt.show()
