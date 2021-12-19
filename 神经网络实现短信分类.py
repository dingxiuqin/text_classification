import matplotlib.pyplot as plt
import pylab as mpl
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from usr_data_process import usr_data_process
from data_process import data_process
import numpy as np

adata, data_after_stop, labels = data_process()
usr_adata, usr_data_after_stop, usr_labels = usr_data_process()
a = adata.values
a = a.reshape(len(adata), 1)

b = labels.values
b = b.reshape(1, len(labels))

x_tr = np.array(a)
y_tr = np.array(b)

x_te = np.array(usr_adata.values.reshape(len(usr_adata), 1))
y_te = np.array(usr_labels.values.reshape(1, len(usr_labels)))

# 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_features=100)
# 该类会统计每个词语的tf-idf权值
tf_idf_transformer = TfidfTransformer()

# 将文本转为词频矩阵并计算tf-idf
# 检查array的形状.如果参数 fit_transform 是字符串数组，它必须是一维数组. (也就是说，array.shape的格式必须为(n,).)例如，如果array具有诸如(n, 1)的形状，则会出现"无属性"错误.
tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_tr.ravel()))
tf_idf1 = tf_idf_transformer.transform(vectorizer.transform(x_te.ravel()))

# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
x_train_weight = tf_idf.toarray()
x_train_weight = x_train_weight.T

x_test_weight = tf_idf1.toarray()
x_test_weight = x_test_weight.T


def cost(w, b, X, Y, learn_rate, A, num_ite):
    cost = []
    for i in range(num_ite):
        m = X.shape[1]
        Z = np.dot(w.T, x_train_weight) + b
        A = G(Z)
        J = (-np.dot(Y, np.log(A).T) - np.dot((1 - Y), np.log(1 - A).T)) / m
        dZ = A - Y
        dw = (np.dot(X, (dZ.T))) / m
        db = (np.sum(dZ)) / m
        w = w - learn_rate * dw
        b = b - learn_rate * db
        if i % 100 == 0:
            cost.append(J)
    return w, b, cost, A


def G(z):
    a = 1 / (1 + np.exp(-z))
    return a


def predict(w, b, X):
    k = np.dot(w.T, X) + b
    Y_pre = G(k)
    return Y_pre


# 画图
def plot(cost1, cost2, cost3, num_ite):
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号’-‘显示为方块的问题
    k1 = np.arange(0, num_ite / 100)
    costs1 = np.squeeze(cost1)  # squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉,如把维度[1,1,10]变成[10，]

    k2 = np.arange(0, num_ite / 100)
    costs2 = np.squeeze(cost2)

    k3 = np.arange(0, num_ite / 100)
    costs3 = np.squeeze(cost3)
    plt.figure()
    plt.plot(k1*100, costs1, label='学习率为5的曲线', linestyle=':')
    plt.plot(k2*100, costs2, label='学习率为1.5的曲线', linestyle='--')
    plt.plot(k3*100, costs3, label='学习率为1的曲线', linestyle='-.')
    plt.legend()
    plt.title('不同学习率下的误差')
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.show()


if __name__ == "__main__":
    # 将w随机初始化为很小的随机值，注意这里不能将w初始化为0，第一个隐藏层中的每个神经元节点将执行相同的计算。 所以即使经过多次梯度下降迭代后，层中的每个神经元节点都会计算出与其他神经元节点相同的东西。
    w = np.random.randn(100).reshape(100, 1)*0.01
    b = 0
    num_ite = 2000 # 迭代次数
    learning_rate1 = 5 # 学习率
    w1, b1, cost1, A1 = cost(w, b, x_train_weight, y_tr, learning_rate1, 0, num_ite) # A的初始值为0

    learning_rate2 = 1.5
    w2, b2, cost2, A2 = cost(w, b, x_train_weight, y_tr, learning_rate2, 0, num_ite)

    learning_rate3 = 1
    w3, b3, cost3, A3 = cost(w, b, x_train_weight, y_tr, learning_rate3, 0, num_ite)

    Y_pre = predict(w2, b2, x_test_weight)
    Y_pre[Y_pre >= 0.5] = 1  # 当假设函数值大于0.5时，预测图片为猫，即为1
    Y_pre[Y_pre < 0.5] = 0  # 当假设函数值小于0.5时，预测图片不为猫，即为0
    for i in range(len(cost1)):
        print('迭代次数为', i * 100, '误差为', cost1[i])

    print('训练集准确度', format(100 - np.mean(np.abs(A1 - y_tr)) * 100), "%")
    print('测试集准确度', format(100 - np.mean(np.abs(Y_pre - y_te)) * 100), "%")

    plot(cost1, cost2, cost3, num_ite)

