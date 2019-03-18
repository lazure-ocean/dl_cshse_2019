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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MASKED_token = 2
MAX_LENGTH = 42

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOSTOKEN": 0, "EOSTOKEN": 1, "MASKEDTOKEN": 2}
        self.index2word = {0: "SOSTOKEN", 1: "EOSTOKEN", 2: "MASKEDTOKEN"}
        self.word2count = {"SOSTOKEN": 0, "EOSTOKEN": 0, "MASKEDTOKEN": 0}
        
        self.n_words = 3  # Count SOS and EOS and Masked token

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
            
def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to
    https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):  # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = " ".join(s.split()[:40])
    return s

def readLang(dataset_title):
    """
    Args:
        dataset_title: either 'imdb' or 'ptb'
    """
    print("Reading lines...")
    if dataset_title == 'imdb':
        train = imdb_dataset(train=True, directory='../data/')
        # Read the dataset and split into lines
        lines = [train[ind]['text'].strip() for ind, doc in enumerate(train)]
        # Normalize lines
        lines = [' '.join(["SOSTOKEN", normalizeString(s), "EOSTOKEN"]) for s in lines]
        lang = Lang(dataset_title)
    elif dataset_title == 'ptb':
        raise NotImplementedError
    return lang, lines

def prepareData(dataset_title):
    lang, lines = readLang(dataset_title)
    print("Read %s sentence pairs" % len(lines))
    print("Counting words...")
    for l in lines:
        lang.addSentence(l)
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, lines

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

class pretrainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        output = self.fc(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
    
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
    
    hidden_size = 64
    train_iters = 10
    dataset = 'imdb'
    lang_filename = './data/' + dataset + '_lang.pkl'
    if os.path.exists(lang_filename):
        with open(lang_filename, 'rb') as file:
            (lang, lines) = pkl.load(file)
    else:
        lang, lines = prepareData(dataset)
        with open(lang_filename, 'wb') as file:
            pkl.dump((lang, lines), file)
    
    #print(random.choice(imdb_lines))
    
    
    model_filename = './pretrained/pretrained_lstm_' + dataset + '_' + str(hidden_size) + '_' + str(train_iters) + '.pkl'
    lstm = pretrainLSTM(lang.n_words, hidden_size).to(device)
    print('using hidden_size=' + str(hidden_size))
    trainIters(lstm, lang, 
               lines, 
               train_iters, 
               print_every=train_iters // 20 + 1, 
               plot_every=train_iters // 50 + 1)
    with open(model_filename, 'wb') as file:
        pkl.dump(lstm, file)
    