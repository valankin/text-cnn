import pandas as pd
import numpy as np
np.random.seed(2)
import time
import sys
import pickle as pkl
from gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split
from data_helpers import pad_sentences,build_vocab,build_input_data

weights_path = '/data/moy_obossanniy_diplom/text-classification-datasets/data/model_weights/'
data_path = '/data/moy_obossanniy_diplom/text-classification-datasets/data/processed/'

#CHANGE DIS
dataset_name = 'mr.all.csv'
weights_name = 'mr'
emb_type = 'w2v_sg'
dataset_path = data_path + dataset_name


def make_text_matrix(sentences):
    sents = []
    for i in range(len(sentences)):
        sentence = sentences[i:i+1]
        text = sentence['text'].values[0]
        if type(text) == unicode or type(text) == str:
            parts = text.split()
            if len(parts) >= 5:
                sents.append(sentence)
    return pd.concat(sents,axis=0)


def preencode(df):
    sentences =  make_text_matrix(df)
    s = [x.split() for x in sentences['text'].values]
    l = sentences['target'].values
    sentences_padded = pad_sentences(s)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, l, vocabulary)
    return x,y,vocabulary,vocabulary_inv


def encode_some_shit(df,weights):
    x,y,vocabulary,vocabulary_inv = preencode(df)
    vectors = []
    vectors = weights[x]
    return vectors,x,y,vocabulary,vocabulary_inv


def get_dummies(y):
    n_classes = len(set(y))
    n_obj = len(y)
    dummies = np.ndarray((n_obj,n_classes),dtype=np.float32)

    for i in range(n_obj):
        dummies[i] = [0.]*n_classes
        j = y[i]
        dummies[i][j] = 1.
                
    return dummies


def get_data():
    print("Loading data...")
    dataset = pd.read_csv(dataset_path)

    print("Loading weights...")
    weights = pkl.load(open(weights_path + weights_name,'r'))
    embedding_weights = weights[emb_type]
    embedding_weights = np.array(embedding_weights)[0]

    print("Encoding data...")
    vectors,x,y,vocabulary,vocabulary_inv = encode_some_shit(dataset,embedding_weights)
    
    return vectors,x,get_dummies(y),vocabulary,vocabulary_inv,embedding_weights
