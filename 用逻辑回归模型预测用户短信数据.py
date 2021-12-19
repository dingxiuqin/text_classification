from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from data_process import data_process
from usr_data_process import usr_data_process

adata, data_after_stop, labels = data_process()
usr_adata, usr_data_after_stop, usr_labels = usr_data_process()
data_tr, data_te, labels_tr, labels_te = train_test_split(adata, labels, test_size=0.2)

x_train = data_tr
y_train = labels_tr
x_test = data_te
y_test = labels_te

# 构造向量
vector = CountVectorizer(analyzer='char', max_features=None, lowercase=False)
vector.fit(x_train)
vector.fit(usr_adata)

model1 = LogisticRegression()
model1.fit(vector.transform(x_train), y_train)
model1.score(vector.transform(x_test), y_test)
m = list(usr_labels)                             # 短信的实际类别
n = list(usr_adata)                              # 短信分词后的内容
k = model1.predict(vector.transform(usr_adata))  # 模型预测的短信的类别
t = 0
for i in range(len(m)):
    if m[i] == k[i]:
        t = t + 1
print("预测用户短信数据的准确率为", t/20)
q = []
w = []
for i in range(len(m)):
    if m[i] == 1:
        q.append('实际为垃圾短信')
    elif m[i] == 0:
        q.append('实际为正常短信')
for i in range(len(k)):
    if k[i] == 1:
        w.append('预测为垃圾短信')
    elif k[i] == 0:
        w.append('预测为正常短信')
z = []
for i in range(len(m)):
    z.append([q[i],w[i],n[i]] )
z = pd.Series(z)

print(z)
