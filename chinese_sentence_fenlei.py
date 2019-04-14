#! -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from M_H_Attention import Attention
from M_H_Attention  import Position_Embedding
import numpy as np
import nltk  #用来分词
import collections  #用来统计词频
from sklearn.model_selection import train_test_split
import codecs
import jieba
from  To_utf_8 import convert
# with open('comments_words_sample.txt','r',encoding='utf-8') as f:
#     for line in f:
#         label, sentence = line.strip().split(',',1)
#         words = nltk.word_tokenize(sentence.lower())
#         if len(words) > maxlen:
#             maxlen = len(words)
#         for word in words:
#             word_freqs[word] += 1
#         num_recs += 1
# print('max_len ',maxlen)
# print('nb_words ', len(word_freqs))
max_features = 20000
maxlen = 80#句子最大长度
batch_size = 32
word_freqs = collections.Counter()  #词频
num_recs = 100 # 样本数
i=0
vocab_size = min(max_features, len(word_freqs)) + 2#词汇大小
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(max_features))}
# word2index["PAD"] = 0
# word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
path="C:\\Users\\Administrator\\PycharmProjects\\M_H_Attention\\gne0_99\\neg"
num = 0
while num<100:
            name = "%d" % num
            print(name)
            fileName = path + "."+str(name) + ".txt"
            text_file =    codecs.open(fileName, 'r',encoding='UTF-8')
            sens = text_file.readlines()
            sentence = (''.join(sens)).rstrip('\n')
            #print(''.join(sens) )#list转为字符串
            words = jieba.cut(sentence,cut_all=False)  #精确模式
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(3)#(word2index["UNK"])
            X[i] = seqs
            y[i] = 1 #neg标记为1
            i += 1
            num +=1
print(X[0],X[1],X[10],'\n',y[0],y[1],y[10])#,X[100],X[1000],X[2999]y[100],y[1000],y[2999]

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
embeddings = Embedding(max_features, 128)(S_inputs)
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

