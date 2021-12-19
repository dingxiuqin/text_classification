import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba # 结巴分词
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库

# 读取文件
with open('message.txt', encoding='utf-8') as fn: # 打开文件
    string_news = fn.read() # 读出整个文件

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"') # 定义正则表达式匹配模式
string_news = re.sub(pattern, '', string_news) # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(string_news, cut_all = False) # 精确模式分词
object_news = []
with open('stopword.txt', encoding='gbk') as fn1:
    remove_words = fn1.read()

for word in seg_list_exact:                            # 循环读出每个分词
    if word not in remove_words:                       # 如果不在去除词库中
        object_news.append(word)
        # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_news)        # 对分词做词频统计
word_counts_top10 = word_counts.most_common(10)     # 获取
print(word_counts_top10)                             # 输出检查

# 词频展示
mask = np.array(Image.open('wordcloud.jpg')) # 定义词频背景
wc = wordcloud.WordCloud(
    font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式
    mask=mask, # 设置背景图
    max_words=200, # 最多显示词数
    max_font_size=100 # 字体最大值
)

wc.generate_from_frequencies(word_counts) # 从字典生成词云
image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
plt.imshow(wc)                      # 显示词云
plt.axis('off')                     # 关闭坐标轴
plt.show()                          # 显示图像
