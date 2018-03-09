# -*- coding: utf-8 -*-
"""
Simple example using a Dynamic RNN (LSTM) to classify IMDB sentiment dataset.
Dynamic computation are performed over sequences with variable length.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import tensorflow as tf
import numpy as np

def generate_seq(num_samples):
    seq = []
    tar = []
    for _ in range(num_samples):
        rd = np.random.randint(1,50)
        tmp = list(range(rd, rd+10))
        seq += tmp
        tar.append(sum(tmp))

    seq = np.array(seq).reshape((-1, 10, 1))
    tar = np.array(tar).reshape((-1, 1))

    return seq, tar


def main():
    # IMDB Dataset loading
    # train, test, _ = imdb.load_data(path='./imdb.pkl', n_words=10000,
    #                                 valid_portion=0.1)
    # trainX, trainY = train
    # testX, testY = test

    trainX, trainY = generate_seq(20000)
    testX, testY = generate_seq(100)

    # Data preprocessing
    # NOTE: Padding is required for dimension consistency. This will pad sequences
    # with 0 at the end, until it reaches the max sequence length. 0 is used as a
    # masking value by dynamic RNNs in TFLearn; a sequence length will be
    # retrieved by counting non zero elements in a sequence. Then dynamic RNN step
    # computation is performed according to that length.
    # trainX = pad_sequences(trainX, maxlen=100, value=0.)
    # testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    # trainY = to_categorical(trainY, 2)
    # testY = to_categorical(testY, 2)

    # Network building
    net = tflearn.input_data([None, 10, 1])
    # Masking is not required for embedding, sequence length is computed prior to
    # the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
    # net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net_lstm, states = tflearn.lstm(net, 128, dropout=0.8, dynamic=True, return_state=True)
    print('s', states)
    net = tflearn.fully_connected(net_lstm, 512, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 64, activation='relu')
    net = tflearn.fully_connected(net, 1, activation='linear')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                             loss='mean_square',  metric='R2')

    # Training
    model = tflearn.DNN(net)
    # model.fit(trainX, trainY, validation_set=(testX, testY), batch_size=128, n_epoch=20, show_metric=True)
    # model.save('./mod.ckpt')

    model.load('./mod.ckpt')

    model2 = tflearn.DNN(states[1], session=model.session)

    print(testX[:2])
    print(model.predict(testX[:2]))

    a = [[[10]]+[[0]]*9]
    print(a)
    print(model.predict(a))
    print(model2.predict(a))


if __name__ == '__main__':
    main()