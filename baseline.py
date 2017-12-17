import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sys
import numpy
import unicodedata
import string

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
toktok = ToktokTokenizer()

test_data_path = sys.argv[1] 
directory_name = sys.argv[2] 
alignment_marker_type = sys.argv[3]


# 윈도우에서 스패인어 보기 위해.
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 대표하는 alignment markers = 이 단어가 후속 문장에 포함되면 alignment 다 라고 생각가능한 마커만 뽑음.
def marker_set(x):
    return {
        '1': ['el', 'la', 'los', 'las', 'lo'], 
        '2': ['el', 'la', 'los', 'las', 'lo', 'sí', 'bien', 'si', 'mucho', 'no'],
        '3': ['el', 'la', 'los', 'las', 'lo', 'sí', 'bien', 'si', 'mucho', 'no', 'con', 'a', 'en','por', 'hay'],
        '4': ['el', 'la', 'los', 'las', 'lo', 'sí', 'bien', 'si', 'mucho', 'no', 'con', 'a', 'en','por', 'hay', 'siempre', 'nunca', 'y','pero','así'],
        '5': [asd],
    }.get(x)

alignment_markers = marker_set(alignment_marker_type)
print([unicodeToAscii(marker) for marker in alignment_markers])


# Test baseline
correct = 0
total = 0
if not os.path.exists('./result'):
    os.makedirs('./result')
if not os.path.exists('./result/'+directory_name):
    os.makedirs('./result/'+directory_name)

with open('./result/'+directory_name+'/baseline_makrer_%d.txt'%int(alignment_marker_type), 'w', encoding ='utf-8') as w, open(test_data_path, 'r', encoding ='utf-8') as test_data:

    for line in test_data.readlines():
        sentence_1, sentence_2, label = line.lower().strip().split('\t')
        word_list = toktok.tokenize(' '.join(nltk.word_tokenize(sentence_2))) # 후속 문장의 단어만 보고 판단.

        predicted = 2 # 기본은 none (2) 이고, marker 가 포함되었을때만 alignment (1) 로 함.
        for candidate in alignment_markers:
            if candidate in word_list:
                predicted = 1
                break
        
        total += 1
        
        if int(predicted) != int(label):
            w.write('label\t' + str(label) +'\t' + 'predicted\t' + str(predicted) +'\t' + sentence_1 +'\t'+ sentence_2 +'\n')
        correct += (int(predicted) == int(label))

    print('Test Accuracy of baseline: %0.2f %%' % (100 * correct / total)) 
    w.write('[Test Accuracy of baseline] : %0.2f %% \n\n' % (100 * correct / total)) 
