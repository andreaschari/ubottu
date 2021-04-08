from __future__ import division
import csv
import numpy as np
import random
import scipy
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *
import pandas as pd

TRAIN_FILE = '/content/ubottu/src/ubuntu_csvfiles/trainset.csv'
VAL_FILE = '/content/ubottu/src/ubuntu_csvfiles/valset.csv'
TEST_FILE = '/content/ubottu/src/ubuntu_csvfiles/testset.csv'

def recall(probas, k, group_size):    
    n_batches = len(probas) // group_size
    n_correct = 0
    for i in xrange(n_batches):
        batch = np.array(probas[i*group_size:(i+1)*group_size])
        #p = np.random.permutation(len(batch))
        #indices = p[np.argpartition(batch[p], -k)[-k:]]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return n_correct / (len(probas) / group_size)

def run(C_vec, R_vec, Y, group_size):
    batch_size = 10
    n_batches = len(Y) // 10
    probas = []
    YY = []
    for i in xrange(n_batches):
        if i % 10000 == 0:
            print i
        batch_c = C_vec[i*batch_size:(i+1)*batch_size][:group_size]
        batch_r = R_vec[i*batch_size:(i+1)*batch_size][:group_size]
        batch_y = Y[i*batch_size:(i+1)*batch_size][:group_size]
        YY.append(batch_y)
        probas += [1 - cosine(batch_c[0].toarray(), r.toarray()) for r in batch_r]
    for k in [1, 2, 5]:
        if k < group_size:
            print 'recall@%d: ' % k, recall(probas, k, group_size)
    probas = np.array(probas)
    pred = np.zeros(probas.shape)
    pred[probas > 0.5] = 1
    pred[probas <= 0.5] = 0
    YY = np.concatenate(YY)
    print "Y=1: ", np.sum(pred)
    print classification_report(YY, pred)
    
def load_data(dataframe):
    '''
    return lists of context, response and flag
    '''
    return  dataframe['context'].to_list(), dataframe['response'].to_list(), list(map(int, dataframe['flag'].to_list()))

val_file_df = pd.read_csv(VAL_FILE, names=['context', 'response', 'flag'])
test_file_df = pd.read_csv(TEST_FILE,  names=['context', 'response', 'flag'])
print 'read csv files'

val_file_df = val_file_df.fillna(' ')
test_file_df = test_file_df.fillna(' ')

val_C, val_R, val_Y = load_data(val_file_df)
print 'loaded val data!'
test_C, test_R, test_Y = load_data(test_file_df)
print 'loaded test data!'

chunk = 0

for train_file_chunk in pd.read_csv(TRAIN_FILE,  names=['context', 'response', 'flag'], chunksize=50000) :
    # replace empty values with space
    train_file_chunk = train_file_chunk.fillna(' ')
    train_C, train_R, train_Y = load_data(train_file_chunk)
    print 'loaded train file batch ', chunk  
    
    vectorizer = TfidfVectorizer()
    print 'training tf-idf baseline'
    vectorizer.fit(train_C+train_R+val_C+val_R)
    
    C_vec = vectorizer.transform(test_C)
    R_vec = vectorizer.transform(test_R)
    Y = np.array(test_Y)
    print 'running evaluation'
    for group_size in [2, 10]:
        run(C_vec, R_vec, Y, group_size)

    chunk += 1
print 'DONE!'
