from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import os
import random
import pickle as pkl
import math
from time import time


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torchnlp.datasets import imdb_dataset
from torchnlp.datasets import penn_treebank_dataset

from models import EncoderRNN, AttnDecoderRNN, pretrainLSTM
from data_preparation import cachePrepareData, Lang
import maskmle
from maskmle import tensorsForTrain
from models import EncoderRNN, AttnDecoderRNN, pretrainLSTM
from data_preparation import cachePrepareData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 42
SOS_token = 0
EOS_token = 1
MASKED_token = 2    

def evaluate(encoder, decoder, input_lang, input_tensor, target_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = input_tensor[0]
        decoder_hidden = encoder_hidden
        decoded_words = ['SOSTOKEN']
        perplexity = 0

        for di in range(max_length):        
            
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)      
            #loss += criterion(decoder_output, target_tensor[di + 1])
            #print(decoder_output.shape, target_tensor[di + 1])
            perplexity -= decoder_output[0, int(target_tensor[di + 1])]
            
            if input_tensor[di + 1].item() == MASKED_token:
                token_sample = torch.multinomial(torch.exp(decoder_output), 1)
                decoder_input = token_sample.squeeze().detach()
                decoded_words.append(input_lang.index2word[decoder_input.item()].upper())
            else:
                decoder_input = input_tensor[di + 1]
                decoded_words.append(input_lang.index2word[decoder_input.item()])
            
            if input_tensor[di + 1].item() == EOS_token:
                perplexity = perplexity / (di + 1)
                break
                
        return torch.exp(perplexity).item()
    
def evaluateRandomly(encoder, decoder, input_lang, input_lines, n=10, is_present=0.5):
    perplexity = 0
    for i in range(n):
        sentence = random.choice(input_lines)   
        pair = tensorsForTrain(input_lang, sentence, is_present=is_present)
        perplexity += evaluate(encoder, decoder, input_lang, pair[0], pair[1])
    return perplexity / n
        
def main():
    dataset = 'imdb'
    hidden_size = 325
    train_iters = 40
    pretrain_train_iters = 40
    lang, lines = cachePrepareData(dataset)

    PATH = './pretrained/'
    pretrained_filename = PATH + 'pretrained_lstm_' + dataset + '_' + str(hidden_size) + '_' + str(pretrain_train_iters) + '.pt'
    
    model_filename = 'maskmle_' + dataset + '_' + str(hidden_size) + '_' + str(train_iters) + '.pt'
    
    encoder1 = EncoderRNN(lang.n_words, hidden_size).to(device)
    encoder1.load_state_dict(torch.load(PATH + 'e_' + model_filename))
    
    attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words, dropout_p=0.1).to(device)
    attn_decoder1.load_state_dict(torch.load(PATH + 'd_' + model_filename))
    print(evaluateRandomly(encoder1, attn_decoder1, lang, lines, 20, 0.5))

if __name__ == "__main__":
    main()