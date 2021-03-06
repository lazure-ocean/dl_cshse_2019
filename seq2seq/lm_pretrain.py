from __future__ import unicode_literals, print_function, division
from io import open
from time import time
import unicodedata
import string
import re
import os
import random
import math
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torchnlp.datasets import imdb_dataset
from torchnlp.datasets import penn_treebank_dataset

from gensim.models import Word2Vec
from data_preparation import cachePrepareData, Lang
from models import EncoderRNN, AttnDecoderRNN, pretrainLSTM, count_parameters
from word2vec_embeddings import get_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MASKED_token = 2
MAX_LENGTH = 42

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    #indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsForTrain(lang, sentence):
    # mask = generate_mask(len(sentence))
    target_tensor = tensorFromSentence(lang, sentence)
    # transformed_sentence = " ".join(transform_input_with_is_missing_token(sentence.split(), mask))
    #input_tensor = tensorFromSentence(lang, transformed_sentence)
    return target_tensor # , target_tensor

def indexFromTensor(lang, decoder_output):
    return decoder_output.max(0)[1]
    
MAX_LENGTH = 42 # max(map(lambda x: len(x.split()), imdb_lines)) == 2516

def train(input_tensor, model, model_optimizer, criterion, max_length=MAX_LENGTH):
    #c_ = time()
    model_hidden = model.initHidden()

    model_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    
    model_outputs = torch.zeros(max_length, model.input_size, device=device)

    loss = 0

    for ei in range(input_length - 1):
        model_output, model_hidden = model(
            input_tensor[ei], model_hidden)
        #print(model_output, input_tensor.shape, input_tensor[0].shape)
        loss += criterion(model_output[0], input_tensor[ei + 1])
        model_outputs[ei] = model_output[0]
    
    loss.backward()

    
    model_optimizer.step()

    return loss.item() / input_length

def test(input_tensor, model, criterion, max_length=MAX_LENGTH):
    model_hidden = model.initHidden()

    input_length = input_tensor.size(0)
    
    model_outputs = torch.zeros(max_length, model.input_size, device=device)

    loss = 0

    for ei in range(input_length - 1):
        model_output, model_hidden = model(
            input_tensor[ei], model_hidden)
        
        loss += criterion(model_output[0], input_tensor[ei + 1])
        model_outputs[ei] = model_output[0]

    return loss.item() / input_length

def trainIters(model, lang, lines, n_iters, print_every=1000, plot_every=100, test_every=1, learning_rate=0.01):
    #start = time.time()
    start = time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_loss_val = 0  # Reset every print_every
    plot_loss_val = 0  # Reset every plot_every

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    training_sentences = [tensorFromSentence(lang, lines[i]) for i in range(n_iters)]
    test_sentences = [tensorFromSentence(lang, lines[n_iters + i]) for i in range(n_iters // test_every)]
    
    criterion = nn.CrossEntropyLoss() 
    test_count = 0
    for iter_ in range(1, n_iters + 1):
        #c_ = time()
        input_tensor = training_sentences[iter_ - 1]

        loss = train(input_tensor, model,
                     model_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        
        if iter_ % test_every == 0:
            test_tensor = test_sentences[iter_ // test_every - 1]
            loss = test(input_tensor, model, criterion)
            print_loss_val += loss
            plot_loss_val += loss
            test_count += 1

        
        if iter_ % print_every == 0:
            print_loss_avg_val = print_loss_val / test_count
            test_count = 0
            print_loss_val = 0
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) train: %.4f, val: %.4f' % (timeSince(start, iter_ / n_iters),
                                         iter_, iter_ / n_iters * 100, print_loss_avg, print_loss_avg_val))

        if iter_ % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    
    #showPlot(plot_losses)
    return plot_losses

#def generate()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

if __name__ == "__main__":
    
    hidden_size = 325
    train_iters = 10
    dataset = 'imdb'
    lang, lines = cachePrepareData(dataset)
    
    model_filename = ''.join(['./pretrained/pretrained_lstm_', 
                              dataset, '_', 
                              str(hidden_size), '_', 
                              str(train_iters), '.pt'])


    # w2v_model = Word2Vec.load(''.join(["word2vec_", str(hidden_size), ".model"]))
    w2v_vectors = get_embeddings(lang, lines, hidden_size)
    w2v_vectors = torch.from_numpy(w2v_vectors).float()
    lstm = pretrainLSTM(lang.n_words, hidden_size).to(device)
    print('lstm initialized')
    print("Total number of trainable parameters without word2vec:", count_parameters(lstm))
    def copy_embedding(layer, vectors):
        return nn.Embedding(layer.num_embeddings, 
                                  layer.embedding_dim).from_pretrained(vectors)
    lstm.embedding = copy_embedding(lstm.embedding, w2v_vectors)
    print("Total number of trainable parameters with word2vec:", count_parameters(lstm))
    
    print('using hidden_size=' + str(hidden_size), ' train_iters = ', train_iters)
    trainIters(lstm, lang, 
               lines, 
               train_iters, 
               print_every=train_iters // 20 + 1, 
               plot_every=train_iters // 50 + 1)
    
    # with open(model_filename, 'wb') as file:
    #    pkl.dump(lstm, file)
    torch.save(lstm.state_dict(), model_filename)    