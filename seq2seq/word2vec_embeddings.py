# inspired with https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

from io import open
from time import time
import unicodedata
import string
import os

import pickle as pkl

import numpy as np


from torchnlp.datasets import imdb_dataset
from torchnlp.datasets import penn_treebank_dataset

from data_preparation import cachePrepareData, Lang

import multiprocessing
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from gensim.models import Word2Vec

def train_word2vec(lang, lines, size, PATH=None):
    
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    
    w2v_model = Word2Vec(min_count=1,
                     window=4,
                     size=size,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

    t = time()
    
    w2v_model.build_vocab(lines, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    
    t = time()
    w2v_model.train(lines, total_examples=w2v_model.corpus_count, epochs=5, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.init_sims(replace=True)
    print('w2v: ', w2v_model.wv)
    if PATH is None:
        w2v_model.save("".join(["word2vec_", str(size), ".model"]))
    else:
        w2v_model.save(PATH)
        
def get_embeddings(lang, lines, size=325):
    lang_filename = "".join(["word2vec_", str(size), ".model"])
    if not os.path.exists(lang_filename):
        lines = [line.split() for line in lines] + [['MASKEDTOKEN']]
        train_word2vec(lang, lines, size)
    w2v_model = Word2Vec.load(lang_filename)
    w2v_model['SOSTOKEN']
    bow = [lang.index2word[key] for key in sorted(lang.index2word.keys())]
    return w2v_model[bow]
    
def main():
    hidden_size = 325
    dataset = 'imdb'
    lang, lines = cachePrepareData(dataset)
            
    lines = [line.split() for line in lines]
    train_word2vec(lang, lines, hidden_size)

if __name__ == '__main__':
    main()