import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import sys
import numpy

from utils import *
from models import *

# Hyper Parameters
max_sequence_length = 30
max_vocabulary_size = 25000
embedding_size = 300
hidden_size = int(sys.argv[5]) # 256
num_layers = 2
num_classes = 2
batch_size = 1
num_epochs = 30
learning_rate = 0.001
dropout_rate = float(sys.argv[6]) #0

train_data_path = 'train_WD_0.10_aug_train_twitter.txt' 
unlabeled_data_path = sys.argv[1]  # preprocessed_10000unlabeled.txt
directory_name = sys.argv[2] #'171218marker'
accuracy = float(sys.argv[3]) # 81.00
model_name = sys.argv[4] #'CNN'

word_to_ix, ix_to_word, vocab_size = make_or_load_dict(train_data_path, character=False)

def model(x):
    return {
        'BiLSTM': BiLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate),
        'CNN': CNN(vocab_size, embedding_size, num_classes, dropout_rate, kernel_num=hidden_size),
        'Cha_CNN_LSTM': Cha_CNN_LSTM(vocab_size, embedding_size, num_classes, dropout_rate, kernel_num =hidden_size),
        'Siamese_BiLSTM': Siamese_BiLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate),
        'Siamese_CNN': Siamese_CNN(vocab_size, num_classes, embedding_size, kernel_num = hidden_size),
    }.get(x)

model = model(model_name)
model.cuda()
model.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_Acc%0.2f_best_valid_accuracy.pkl' % (model_name, hidden_size, dropout_rate, accuracy)))
#model.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_Acc%0.2f.pkl' % (model_name, hidden_size, dropout_rate, accuracy)))
print(model)

# Predict whether tweet-reply pair is alignment or not
correct = 0
total = 0
if not os.path.exists('./result'):
    os.makedirs('./result')
if not os.path.exists('./result/'+directory_name):
    os.makedirs('./result/'+directory_name)

with open(unlabeled_data_path,'r', encoding = 'utf-8') as tweet, \
     open('./result/'+directory_name+'/Align_%s_hid%d_D%0.2f_Acc%0.2f.txt'%(model_name, hidden_size, dropout_rate, accuracy), 'w', encoding ='utf-8') as Align, \
     open('./result/'+directory_name+'/None_%s_hid%d_D%0.2f_Acc%0.2f.txt'%(model_name, hidden_size, dropout_rate, accuracy), 'w', encoding ='utf-8') as none:

    count = 0
    for sentences in tweet.readlines():
        count += 1
        if count % 100 == 0:
            print('pair_count', count)

        sentence_1_origin, sentence_2_origin, _ = sentences.lower().strip().split('\t')
        
        if model_name in ['BiLSTM','CNN','Cha_CNN_LSTM']:
            sentence_concat = (sentence_1_origin+' '+sentence_2_origin).split()
            sentence_concat = prepare_sequence(sentence_concat, word_to_ix)

            outputs = model(Variable(sentence_concat.view(batch_size, -1)).cuda(), train=False)
            _, predicted = torch.max(outputs.data, 1) # 두번째 아웃풋 값은 argmax 를 반환

        elif model_name in ['Siamese_BiLSTM','Siamese_CNN']:
            sentence_1 = sentence_1_origin.split()
            sentence_1 = prepare_sequence(sentence_1, word_to_ix)

            sentence_2 = sentence_2_origin.split()
            sentence_2 = prepare_sequence(sentence_2, word_to_ix)

            output = model(Variable(sentence_1.view(batch_size, -1)).cuda(), Variable(sentence_2.view(batch_size, -1)).cuda(), train=False)
            _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환
        
        
        if predicted.cpu().numpy() == 0:
            Align.write(sentence_1_origin+'\t'+sentence_2_origin+'\t'+'1'+'\n') # alignment pair 를 원문 그대로 저장.
        else:
            none.write(sentence_1_origin+'\t'+sentence_2_origin+'\t'+'2'+'\n') # non-alignment pair 를 원문 그대로 저장.
            

# 코퍼스의 각 단어별 빈도를 센 뒤, 빈도순으로 나열해서 저장.
# from https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
from collections import Counter
counts = Counter()
align_count = 0
with open('./result/'+directory_name+'/Align_%s_hid%d_D%0.2f_Acc%0.2f.txt'%(model_name, hidden_size, dropout_rate, accuracy), 'r', encoding = 'utf-8') as r_align, \
    open('./result/'+directory_name+'/%s_Align_marker_%0.2f.txt'%(model_name, accuracy), 'w', encoding = 'utf-8') as w_align:
    for line in r_align.readlines():
        counts.update(line.rstrip().split())
        align_count += 1

    weights = {word: count/align_count for word, count in counts.items()}
    print('align_count', align_count)
    for key, count in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        w_align.write(key+'\t'+str(count)+'\n')


counts_none = Counter()
none_count = 0
with open('./result/'+directory_name+'/None_%s_hid%d_D%0.2f_Acc%0.2f.txt'%(model_name, hidden_size, dropout_rate, accuracy), 'r', encoding = 'utf-8') as r_none, \
    open('./result/'+directory_name+'/%s_None_marker_%0.2f.txt'%(model_name, accuracy), 'w', encoding = 'utf-8') as w_none:
    for line in r_none.readlines():
        counts_none.update(line.rstrip().split())
        none_count +=1

    weights_none = {word: count/none_count for word, count in counts_none.items()}
    print('none_count', none_count)

    for key, count in sorted(weights_none.items(), key=lambda x: x[1], reverse=True):
        w_none.write(key+'\t'+str(count)+'\n')
