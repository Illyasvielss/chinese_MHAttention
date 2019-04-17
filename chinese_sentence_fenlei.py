#! -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence
from M_H_Attention import Attention
from M_H_Attention  import Position_Embedding
import numpy as np
import collections  #用来统计词频
from sklearn.model_selection import train_test_split
from cutWord import read_file_cut

maxlen = 0  #句子最大长度
word_freqs = collections.Counter()  #词频
num_recs = 105 # 样本数
num =0
file_path_neg = "C:\\Users\\Administrator\\PycharmProjects\\M_H_Attention\\neg0_99\\neg."
while num<100:
    name = "%d" % num
    # print(name)
    fileName = file_path_neg + str(name) + ".txt"
    words = read_file_cut(fileName, num)  # ['服务态度', '前台', '接待', '受过', '培训', '礼貌', '接待', '几个', '客人']
    # print(num,'lennn',len(words),words)
    if len(words) > maxlen:
        maxlen = len(words)
        # print(num,'lennn',len(words),words)
    for word in words:
        word_freqs[word] += 1
        # print('www', word)
    num += 1
num=0
file_path_pos = "C:\\Users\\Administrator\\PycharmProjects\\M_H_Attention\\ChnSentiCorp_htl_unba_10000\\pos\\pos."
while num<5:
    name = "%d" % num
    fileName = file_path_pos + str(name) + ".txt"
    words = read_file_cut(fileName, num)
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num += 1
print('max_len ', maxlen)#37
print('nb_words ', len(word_freqs))#54
#建立两个 lookup tables，分别是 word2index 和 index2word，用于单词和数字转换
MAX_FEATURES=1350
MAX_SENTENCE_LENGTH = 200
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2#VOCABULARY_SIZE 包含训练数据中按词频从大到小排序后的前 2000 个单词，外加一个伪单词 UNK 和填充单词 0。
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
#print('www',word2index){'\r\n': 2, '客人': 3, '接待': 4, '酒店': 5, '前台': 6, '标准间': 7, '入住': 8, '服务态度': 9, '受过': 10, '培训': 11, '礼貌': 12, '几个': 13, '大堂': 14, '副理': 15, '辩解': 16, '没完': 17, '总经理': 18, '电话': 19, '投诉': 20, '亏心事': 21, '跟本': 22, '不用': 23, '好久': 24, '评价': 25, '记得': 26, '火车站': 27, '超级': 28, '韩日': 29, '旅游团': 30, '服务': 31, '冷淡': 32, '两个': 33, '人住': 34, '一张': 35, '房卡': 36, '挑衅': 37, '心情': 38, '宾馆': 39, '反馈': 40, '2008': 41, '17': 42, '提出': 43, '现已': 44, '整改': 45, '希望': 46, '一位': 47, '渤海': 48, '明珠': 49, '满意': 50, '房间': 51}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

batch_size = 32
X = np.empty(num_recs,dtype=list)#[None None None] [0. 0. 0.]
y = np.zeros(num_recs)
num=0
while num<100:
    name = "%d" % num
    #print(name)
    fileName = file_path_neg + str(name) + ".txt"
    words = read_file_cut(fileName, num)  # jieba.cut(sentence,cut_all=False)  #精确模式
    seqs = []
    #print(words)['标准间', '房间', '设施', '陈旧', '建议', '酒店', '标准间', '改善', '\r\n', '\r\n']
    #sss [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for word in words:
        if word in word2index:  # word2index:
             seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    #print('sss', seqs)
    X[num] = seqs
    y[num] = 1  # neg标记为1
    num += 1
i=0
while i<5:
    name = "%d" % i
    #print(name)
    fileName = file_path_pos + str(name) + ".txt"
    words = read_file_cut(fileName, num)
    seqs = []
    for word in words:
        if word in word2index:  # word2index:
             seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    #print('sss', seqs)
    X[num] = seqs
    y[num] = 0  # neg标记为1
    num += 1
    i+=1
print(X[103],'\n',X[11],'\n',X[29],'\n',y[103],'\n',y[11],'\n',y[29])#,X[100],X[1000],X[2999]y[100],y[1000],y[2999]

X = sequence.pad_sequences(X, maxlen=maxlen)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Model
from keras.layers import *

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(MAX_FEATURES, 128)(S_inputs)
embeddings = Position_Embedding()(embeddings) #增加Position_Embedding能轻微提高准确率
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))

