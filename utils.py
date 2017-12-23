import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sys
import numpy
import os.path

import unicodedata
import string
import math

train_data_path = sys.argv[1] #"train_data.txt"
test_data_path = sys.argv[2] #"test_data.txt"


# 윈도우에서 스패인어 보기 위해.
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
#print(unicodeToAscii('Ślusàrski'))

def make_or_load_dict(train_data_path, character=False):
    something_to_ix = 'word_to_ix_' if not character else 'character_to_ix_' # 케릭터래벨은, 데이터 저장시에 케릭터 단위로하면 되도록 해둠.

    if not os.path.exists('./vocab'):
        os.makedirs('./vocab')

    # word_to_ix
    if os.path.isfile('./vocab/'+something_to_ix+train_data_path):
        word_to_ix = {}
        with open('./vocab/'+something_to_ix+train_data_path,'r', encoding='utf-8') as dictionary:
            for line in dictionary.readlines():
                word, idx = line.strip().split('\t')
                word_to_ix[word] = int(idx)

    else:
        word_to_ix = {"unk":0, "endofsentence":1}
        with open(train_data_path, 'r', encoding ='utf-8') as data:
            for line in data.readlines():
                sentence_1, sentence_2, label = line.lower().strip().split('\t')
                word_list = (sentence_1+' '+sentence_2).split()
                for word in word_list:
                    if word not in word_to_ix:
                        word_to_ix[word] = len(word_to_ix)

        with open('./vocab/'+something_to_ix+train_data_path, 'w', encoding ='utf-8') as w:
            for word, idx in word_to_ix.items():
                w.write(word+'\t'+str(idx)+'\n')
    
    # ix to word
    ix_to_word = [ i for i in range(len(word_to_ix))]
    for word, idx in word_to_ix.items():
        ix_to_word[idx] = word

    with open('./vocab/'+something_to_ix+train_data_path[:-4]+'_inverse_dict.txt', 'w', encoding ='utf-8') as w:
        for idx, word in enumerate(ix_to_word):
            w.write(word+'\t'+str(idx)+'\n')

    vocab_size=len(word_to_ix)
    print(vocab_size)  
    return word_to_ix, ix_to_word, vocab_size


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix['unk'] for w in seq]
    #tensor = torch.FloatTensor([idxs]) 
    tensor = torch.LongTensor([idxs]) 
    return tensor


def prepare_label(label, to_ix):
    idxs = to_ix[label]
    tensor = torch.LongTensor([idxs])
    return tensor


# pad sequences from https://docs.google.com/presentation/d/18JvZ5n49tt3-8nhBappz6aaraZMazF06v2YuQxGVKEQ/edit#slide=id.g29bd0a4235_0_107
def pad_sequences(vectorized_seqs, seq_lengths):
   seq_lengths_tensor = torch.LongTensor(seq_lengths)
   seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths_tensor.max())).long()
   for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths_tensor)):
       seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
   return seq_tensor


class AlignmentDataset(Dataset):
    """ Alignment dataset."""
    # Initialize your data, download, etc.
    def __init__(self, data_path, word_to_ix, batch_size):
        data_file = open(data_path, 'r', encoding ='utf-8')
        self.data = data_file.readlines()
        self.len = math.floor(len(self.data)/batch_size)
        self.word_to_ix = word_to_ix
        self.label_to_ix = {"1": 0, "2": 1}
        self.batch_size = batch_size
        #self.label_to_ix = {"Alignment": 0, "None": 1}


    def __getitem__(self, index):
        vectorized_seqs, seq_lengths = [], []
        label_batch = torch.zeros(self.batch_size, 1).long()

        for i in range(self.batch_size):
            self.line = self.data[index*self.batch_size + i]
            sentence_1, sentence_2, label = self.line.lower().strip().split('\t')
            
            self.sentence = (sentence_1+' '+sentence_2).split()
            self.sentence_tensor = prepare_sequence(self.sentence, self.word_to_ix)
            seq_lengths.append(len(self.sentence))
            vectorized_seqs.append(self.sentence_tensor)

            self.label = prepare_label(label, self.label_to_ix)
            label_batch[i] = self.label

        return pad_sequences(vectorized_seqs, seq_lengths), label_batch

    def __len__(self):
        return self.len



class AlignmentDataset_seperate_sent(Dataset):
    """ Alignment dataset."""
    # Initialize your data, download, etc.
    def __init__(self, data_path, word_to_ix, batch_size):
        data_file = open(data_path, 'r', encoding ='utf-8')
        self.data = data_file.readlines()
        self.len = math.floor(len(self.data)/batch_size)
        self.word_to_ix = word_to_ix
        self.label_to_ix = {"1": 0, "2": 1}
        self.batch_size = batch_size
        #self.label_to_ix = {"Alignment": 0, "None": 1}

    def __getitem__(self, index):
        vectorized_seqs_1, vectorized_seqs_2, seq_lengths_1, seq_lengths_2 = [], [], [], []
        label_batch = torch.zeros(self.batch_size, 1).long()

        for i in range(self.batch_size):
            self.line = self.data[index*self.batch_size + i]
            sentence_1, sentence_2, label = self.line.lower().strip().split('\t')

            self.sentence_1 = sentence_1.split()
            self.sentence_1_tensor = prepare_sequence(self.sentence_1, self.word_to_ix)
            seq_lengths_1.append(len(self.sentence_1))
            vectorized_seqs_1.append(self.sentence_1_tensor)

            self.sentence_2 = sentence_2.split()
            self.sentence_2_tensor = prepare_sequence(self.sentence_2, self.word_to_ix)
            seq_lengths_2.append(len(self.sentence_2))
            vectorized_seqs_2.append(self.sentence_2_tensor)

            self.label = prepare_label(label, self.label_to_ix)
            label_batch[i] = self.label

        return pad_sequences(vectorized_seqs_1, seq_lengths_1), pad_sequences(vectorized_seqs_2, seq_lengths_2), label_batch


    def __len__(self):
        return self.len


class AlignmentDataset_cha_cnn(Dataset):
    """ Alignment dataset."""
    # Initialize your data, download, etc.
    def __init__(self, data_path, character_to_ix):
        data_file = open(data_path, 'r', encoding ='utf-8')
        self.data = data_file.readlines()
        self.len = len(self.data)
        self.character_to_ix = character_to_ix
        self.label_to_ix = {"1": 0, "2": 1}
        #self.label_to_ix = {"Alignment": 0, "None": 1}

    def __getitem__(self, index):
        self.line = self.data[index]
        sentence_1, sentence_2, label = self.line.lower().strip().split('\t')
        
        self.sentence = (sentence_1+' '+sentence_2).split()
        self.sentence = [prepare_sequence(list(word), character_to_ix) for word in self.sentence]
        self.label = prepare_label(label, label_to_ix)

        return self.sentence, self.label

    def __len__(self):
        return self.len



class GenerateDataset(Dataset):
    """ Alignment dataset."""
    # Initialize your data, download, etc.
    def __init__(self, data_path, word_to_ix, batch_size):
        data_file = open(data_path, 'r', encoding ='utf-8')
        self.data = data_file.readlines()
        self.len = math.floor(len(self.data)/batch_size)
        self.word_to_ix = word_to_ix
        self.batch_size = batch_size
        
    def __getitem__(self, index):
        vectorized_seqs_1, vectorized_seqs_2, seq_lengths_1, seq_lengths_2 = [], [], [], []
        
        for i in range(self.batch_size):
            self.line = self.data[index*self.batch_size + i]
            sentence_1, sentence_2, label = self.line.lower().strip().split('\t')

            self.sentence_1 = sentence_1.split()
            self.sentence_1_tensor = prepare_sequence(self.sentence_1, self.word_to_ix)
            seq_lengths_1.append(len(self.sentence_1))
            vectorized_seqs_1.append(self.sentence_1_tensor)

            self.sentence_2 = sentence_2.split()
            self.sentence_2_tensor = prepare_sequence(self.sentence_2, self.word_to_ix)
            seq_lengths_2.append(len(self.sentence_2))
            vectorized_seqs_2.append(self.sentence_2_tensor)
            
        return pad_sequences(vectorized_seqs_1, seq_lengths_1), pad_sequences(vectorized_seqs_2, seq_lengths_2)

    def __len__(self):
        return self.len



class ContrastiveLoss(torch.nn.Module):
    """
    Borrowed from: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label.float()) * torch.pow(euclidean_distance, 2) +
                                      (label.float()) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive




######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)