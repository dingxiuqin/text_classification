from datetime import time
from scipy import sparse, io
from sklearn import svm
from sklearn.decomposition import PCA
from usr_data_process import usr_data_process
from sklearn.metrics import classification_report, recall_score, f1_score
import matplotlib.pyplot as plt
import pylab as mpl
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # 导入朴素贝叶斯
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # 导入文本特征提取模块  转换成词频  向量转换成TF-IDF权重矩阵
from sklearn.linear_model import LogisticRegression
from data_process import data_process


def dimensionality_reduction(training_data, test_data):
        pca = PCA(n_components=0.2)
        pca.fit(training_data)
        training_data_transform = pca.transform(training_data)
        test_data_transform = pca.transform(test_data)
        return training_data_transform, test_data_transform



def MultinomialNBmodelwork(n):  # 朴素贝叶斯模型训练
    a = []   # 此数组用于储存每次欠抽样后模型的准确率
    a1 = []  # 此数组用于储存每次欠抽样后模型的召回率
    a2 = []  # 此数组用于储存每次欠抽样后模型的f1值
    for i in range(n):
        adata, data_after_stop, labels = data_process()
        data_tr, data_te, labels_tr, labels_te = train_test_split(adata, labels, test_size=0.2)

        x_train = data_tr
        y_train = labels_tr
        x_test = data_te
        y_test = labels_te

        # 构造向量
        vector = CountVectorizer(analyzer='char', max_features=None, lowercase=False)
        vector.fit(x_train)

        model1 = MultinomialNB()
        model1.fit(vector.transform(x_train), y_train)
        predict_label1 = model1.predict(vector.transform(x_test))
        a.append(model1.score(vector.transform(x_test), y_test))
        a1.append(recall_score(y_test, predict_label1))
        a2.append(f1_score(y_test, predict_label1))
    return a, a1, a2

def GaussianNBmodelwork(n):  # 高斯贝叶斯模型训练
    a = []
    a1 = []
    a2 = []
    for i in range(n):
        adata, data_after_stop, lables = data_process()

        data_tr, data_te, labels_tr, labels_te = train_test_split(adata, lables, test_size=0.2)

        countVectorizer = CountVectorizer()  # 使训练集与测试集的列数相同
        data_tr = countVectorizer.fit_transform(data_tr)
        X_tr = TfidfTransformer().fit_transform(data_tr.toarray()).toarray()  # 训练集TF-IDF权值

        data_te = CountVectorizer(vocabulary=countVectorizer.vocabulary_).fit_transform(data_te)
        X_te = TfidfTransformer().fit_transform(data_te.toarray()).toarray()  # 测试集TF-IDF权值

        model = GaussianNB()
        model.fit(X_tr, labels_tr)
        pre = model.predict(X_te)
        a.append(model.score(X_te, labels_te))
        a1.append(recall_score(labels_te, pre))
        a2.append(f1_score(labels_te, pre))
        # print("预测测试集上新样本的分类:", model.predict(X_te))
        # print("各个种类新闻在测试集新样本上的准确率\n", classification_report(labels_te, a))
    return a, a1, a2



def SVMmodelwork(n):  # 支持向量机模型训练
    a = []
    a1 = []
    a2 = []
    for i in range(n):
        adata, data_after_stop, lables = data_process()
        data_tr, data_te, labels_tr, labels_te = train_test_split(adata, lables, test_size=0.2)

        countVectorizer = CountVectorizer()  # 使训练集与测试集的列数相同
        data_tr = countVectorizer.fit_transform(data_tr)
        X_tr = TfidfTransformer().fit_transform(data_tr.toarray()).toarray()  # 训练集TF-IDF权值

        data_te = CountVectorizer(vocabulary=countVectorizer.vocabulary_).fit_transform(data_te)
        X_te = TfidfTransformer().fit_transform(data_te.toarray()).toarray()  # 测试集TF-IDF权值
        X = X_tr
        y = labels_tr
        #X_tr, X_te = dimensionality_reduction(X_tr, X_te)  # 降维处理,cpu过载
        model = svm.LinearSVC()
        model.fit(X_tr, labels_tr)
        pre = model.predict(X_te)
        a.append(model.score(X_te, labels_te))
        a1.append(recall_score(labels_te, pre))
        a2.append(f1_score(labels_te, pre))
    # print("预测测试集上新样本的分类:", model.predict(X_te))
    # print("各个种类新闻在测试集新样本上的准确率\n", classification_report(labels_te, a))
    return a, a1, a2


def LogisticRegressionmodelwork(n):  # 逻辑回归模型训练
    a = []  # 此数组用于储存每次欠抽样后模型的准确率
    a1 = []
    a2 = []
    for i in range(n):
        adata, data_after_stop, labels = data_process()
        data_tr, data_te, labels_tr, labels_te = train_test_split(adata, labels, test_size=0.2)

        x_train = data_tr
        y_train = labels_tr
        x_test = data_te
        y_test = labels_te

        # 构造向量
        vector = CountVectorizer(analyzer='char', max_features=None, lowercase=False)
        vector.fit(x_train)

        model1 = LogisticRegression(max_iter=1000)
        model1.fit(vector.transform(x_train), y_train)
        predict_label1 = model1.predict(vector.transform(x_test))
        a.append(model1.score(vector.transform(x_test), y_test))
        a1.append(recall_score(y_test, predict_label1))
        a2.append(f1_score(y_test, predict_label1))
    return a, a1, a2

# 以准确率为标准画图
def score(b, b1, b2 , b3 , n):
    k = np.arange(1, n + 1, 1)  # 生成1到n的数组，用来当横坐标
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号’-‘显示为方块的问题
    # 朴素贝叶斯模型召回率分布折线图
    plt.figure()
    plt.plot(k, b, label='朴素贝叶斯模型准确率分布', linestyle=':')
    # 高斯贝叶斯模型召回率分布折线图
    plt.plot(k, b1, label='高斯贝叶斯模型准确率分布', linestyle='--')
    # 支持向量机模型召回率分布折线图
    plt.plot(k, b2, label='支持向量机模型准确率分布', linestyle='-.')
    # 逻辑回归模型召回率分布折线图
    plt.plot(k, b3, label='逻辑回归模型准确率分布', linestyle='-')
    plt.legend()
    plt.title('各类模型准确率')
    plt.xlabel('抽样次数')
    plt.ylabel('准确率')
    plt.show()

# 以召回率为标准画图
def recallscore(b, b1, b2 , b3 , n):
    k = np.arange(1, n + 1, 1)  # 生成1到n的数组，用来当横坐标
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号’-‘显示为方块的问题
    # 朴素贝叶斯模型召回率分布折线图
    plt.figure()
    plt.plot(k, b, label='朴素贝叶斯模型召回率分布', linestyle=':')
    # 高斯贝叶斯模型召回率分布折线图
    plt.plot(k, b1, label='高斯贝叶斯模型召回率分布', linestyle='--')
    # 支持向量机模型召回率分布折线图
    plt.plot(k, b2, label='支持向量机模型召回率分布', linestyle='-.')
    # 逻辑回归模型召回率分布折线图
    plt.plot(k, b3, label='逻辑回归模型召回率分布', linestyle='-')
    plt.legend()
    plt.title('各类模型召回率')
    plt.xlabel('抽样次数')
    plt.ylabel('召回率')
    plt.show()

def f1score(b, b1, b2 , b3 , n):
    k = np.arange(1, n + 1, 1)  # 生成1到n的数组，用来当横坐标
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号’-‘显示为方块的问题
    # 朴素贝叶斯模型召回率分布折线图
    plt.figure()
    plt.plot(k, b, label='朴素贝叶斯模型f1值分布', linestyle=':')
    # 高斯贝叶斯模型召回率分布折线图
    plt.plot(k, b1, label='高斯贝叶斯模型f1值分布', linestyle='--')
    # 支持向量机模型召回率分布折线图
    plt.plot(k, b2, label='支持向量机模型f1值分布', linestyle='-.')
    # 逻辑回归模型召回率分布折线图
    plt.plot(k, b3, label='逻辑回归模型f1值分布', linestyle='-')
    plt.legend()
    plt.title('各类模型f1值')
    plt.xlabel('抽样次数')
    plt.ylabel('f1值')
    plt.show()

# 以准确度为标准画图
if __name__ == "__main__":
    n = 10
    Mb, Mb1, Mb2 = MultinomialNBmodelwork(n)
    Gb, Gb1, Gb2 = GaussianNBmodelwork(n)
    Sb, Sb1, Sb2 = SVMmodelwork(n)
    Lb ,Lb1, Lb2 = LogisticRegressionmodelwork(n)
    print('n次抽样得到的朴素贝叶斯模型平均准确率：', sum(Mb) / n)
    print('n次抽样得到的高斯贝叶斯模型平均准确率：', sum(Gb) / n)
    print('n次抽样得到的支持向量机模型平均准确率：', sum(Sb) / n)
    print('n次抽样得到的逻辑回归模型平均准确率：', sum(Lb) / n)
    print('n次抽样得到的朴素贝叶斯模型平均召回率：',sum(Mb1)/n)
    print('n次抽样得到的高斯贝叶斯模型平均召回率：', sum(Gb1) / n)
    print('n次抽样得到的支持向量机模型平均召回率：', sum(Sb1) / n)
    print('n次抽样得到的逻辑回归模型平均召回率：', sum(Lb1) / n)
    print('n次抽样得到的朴素贝叶斯模型平均f1值：',sum(Mb2)/n)
    print('n次抽样得到的高斯贝叶斯模型平均f1值：', sum(Gb2) / n)
    print('n次抽样得到的支持向量机模型平均f1值：', sum(Sb2) / n)
    print('n次抽样得到的逻辑回归模型平均f1值：', sum(Lb2) / n)

    recallscore(Mb1, Gb1, Sb1, Lb1, n)
    #score(Mb, Gb, Sb, Lb, n)
    #f1score(Mb2, Gb2, Sb2, Lb2, n)