import pandas as pd
import re
import jieba


def usr_data_process(message='userdata.csv'):
    data = pd.read_csv(message, header=None, index_col=0, encoding='utf-8')  # 把数据读取进来

    data.columns = ['label', 'message']  # 列名赋值->标签 内容
    a = data[data['label'] == 0].sample(10)  # 反例正常， sample函数功能为从序列a中随机抽取n个元素，并将5个元素生以list形式返回。
    b = data[data['label'] == 1].sample(10)  # 正例垃圾
    data_new = pd.concat([a, b], axis=0)  # 纵向拼接
    data_dup = data_new['message'].drop_duplicates()  # 短信去重

    def resub(data):  # 将字符‘x’替换为空
        return re.sub('x', '', data)

    def cut(data):  # 分词操作
        return jieba.lcut(data)

    def after_stop(data):  # 去停用词操作
        for i in data:
            if i in stopWords:
                data.remove(i)
        return data

    def joinin(data):  # 列表拼接
        return ' '.join(data)

    data_resub = data_dup.apply(resub)  # 调用替换函数

    jieba.load_userdict('newdic1.txt')  # 将自定义的词典加入
    data_cut = data_resub.apply(cut)  # 调用分词函数

    # 去除停用词
    stopWords = pd.read_csv('stopword.txt', encoding='GB18030', sep='hahaha', header=None, engine='python')  # 导入停用词,
    # 编码，设置分隔符号
    stopWords = ['≮', '≯', '≠', '≮', ' ', '会', '月', '日', '–'] + list(stopWords.iloc[:, 0])  # 增加的分词与以列表为形式的原有分词拼接起来

    data_after_stop = data_cut.apply(after_stop)  # 调用去停用词函数

    # 数据预处理函数封装
    labels = data_new.loc[data_after_stop.index, 'label']  # 标签
    adata = data_after_stop.apply(joinin)  # 将列表进行拼接

    return adata, data_after_stop, labels