import os
import re
import numpy as np
import jieba
import random


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def getLangs(lang1, lang2, reverse=False):
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang


def readSougoCorpus():
    filelist = os.listdir(r'C:\YWTraining\translation\transformer\SougoCorpus')
    engfllst = [file for file in filelist if re.match(re.compile('train.en\w*'), file)]
    zhofllst = [file for file in filelist if re.match(re.compile('train.zh\w*'), file)]
    # print(len(engfllst), len
    englines = []
    zholines = []
    dom = os.path.abspath(r'C:\YWTraining\translation\transformer\SougoCorpus')
    curdir = os.getcwd()
    os.chdir(dom)
    for file in engfllst:
        flpth = os.path.join(dom, file)
        # print(flpth)
        fllines = open(file, encoding='utf-8'). \
            read().strip().split('\n')
        for line in fllines:
            englines.append(line)
        # print(type(fllines), len(fllines), len(englines))

    for file in zhofllst:
        fllines = open(file, encoding='utf-8'). \
            read().strip().split('\n')
        for line in fllines:
            # tokenized = jieba.cut(line)
            # line = ' '.join(tokenized)
            zholines.append(line)
        # print(type(fllines), len(fllines), len(zholines))

    os.chdir(curdir)
    print("sentences count before filtering: ", len(zholines), len(englines))
    pairs = list(zip(zholines, englines))
    pairs = [list(p) for p in pairs]
    return pairs


def saveData(pairs, out_name):
    #filterPairs(pairs)
    #print(pairs[0][0], pairs[0][1])
    #print(pairs[2], pairs[3])
    #print(pairs[4], pairs[5])
    x = [p[0] for p in pairs if len(p[1]) < 15]
    y = [p[1] for p in pairs if len(p[1]) < 15]
    print('sentences count after filterPairs method: ', len(x), len(y))
    np.savez(out_name, x_train= x, y_train= y)
    return


def loadData(lang1, lang2, path, reverse=False, max_len=10):
    f = np.load(path)
    #print(f['x_train'])
    x_train = f['x_train']
    y_train = f['y_train']
    tkx_train = []
    for x in x_train:
        tokenized = jieba.cut(x)
        x = ' '.join(tokenized)
        tkx_train.append(x)
    x_train = tkx_train
    pairs = list(zip(x_train, y_train))# 合并列表对应位置的元素，得到元组的列表
    pairs = [p for p in pairs if len(p[0]) < (max_len+1) & len(p[1]) < (max_len+1)]
    # 限制长度时SOS、EOS怎么加进去？批次训练是否一定要限制句子的长度？
    print(len(x_train),len(y_train))

    input_lang, output_lang = getLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    indexes = np.array(indexes)
    return indexes


def arrFromPair(input_lang, output_lang, pair):
    input_arr = np.array(indexesFromSentence(input_lang, pair[0]))
    # print(input_tensor)
    target_arr = np.array(indexesFromSentence(output_lang, pair[1]))
    return input_arr, target_arr


def sentenceFromArray(lang, array):
    words = [lang.index2word[index] for index in array]
    sentence = str(words)
    return sentence


def split_tra_val(pairs, val_prob=0.2):
    val_indices = np.random.choice(len(pairs), round(len(pairs)*val_prob), replace=False)
    tra_indices = np.array(list(set(range(len(pairs))) - set(val_indices)))
    vpairs = np.array(pairs)[val_indices]
    tpairs = np.array(pairs)[tra_indices]
    return tpairs, vpairs



def tbatch_generator(input_lang, output_lang, tpairs, batch_size, n_iters):
    training_pairs = [arrFromPair(input_lang, output_lang, pair) for pair in tpairs]
    np.random.shuffle(training_pairs)
    while True:
        for iter in range(0, n_iters):

            # if iter + batch_size < len(training_pairs)-1:
            #     train_batch = training_pairs[iter: iter + batch_size]
            # else:
            #     train_batch = np.random.choice(training_pairs, batch_size)
            # 删掉的这个做法有错，后面将改成预热阶段不保存模型，按顺序过完所有数据，后一阶段随机抽取批数据并按save_every_n保存模型！
            indices = np.random.choice(range(len(training_pairs)), batch_size)
            train_batch = np.array(training_pairs)[indices]
            # print(batch_size) 50
            # print(np.array(train_batch).shape) (50,2)
            x = np.zeros((batch_size, 10))
            y = np.zeros((batch_size, 10))

            for i in range(batch_size):
                for j in range(len(train_batch[i][0])):
                    x[i][j] = train_batch[i][0][j]
                for k in range(len(train_batch[i][1])):
                    y[i][k] = train_batch[i][1][k]

            yield x, y


def valarr_generator(input_lang, output_lang, vpairs):
    values_for_vp = [arrFromPair(input_lang, output_lang, pair) for pair in vpairs]
    # print(values_for_vp[0])
    vset_size = len(values_for_vp) # 目前是50
    validation_pairs = np.zeros((vset_size, 2, 10))
    for i in range(vset_size):
        for j in range(len(values_for_vp[i][0])):
            validation_pairs[i][0][j] = values_for_vp[i][0][j]
        for k in range(len(values_for_vp[i][1])):
            validation_pairs[i][1][k] = values_for_vp[i][1][k]

    return validation_pairs
