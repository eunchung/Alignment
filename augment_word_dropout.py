import sys
import numpy as np
from sklearn.model_selection import train_test_split

import re

import unicodedata
import string
import math
import random

# 윈도우에서 스패인어 보기 위해.
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


word_dropout_rate = 0.1

output = 'WD_%0.2f_aug_'%word_dropout_rate + sys.argv[1]

number_of_augmented = 5
with open(output, 'a', encoding ='utf-8') as w_augmented,\
	open('./'+sys.argv[1], 'r', encoding='utf-8') as data:
	train_corpus = data.readlines()

	# # 본래 train_data 
	# for train_sent in train_corpus:
	# 	sent1, sent2, label = train_sent.strip().split('\t')
	# 	w_augmented.write(sent1.strip()+' EndOfSentence\t'+sent2.strip()+' EndOfSentence\t'+ label.strip() +'\n')

	# number_of_augmented 만큼 augmented 한 데이터를 추가함.
	for _ in range(number_of_augmented):
		for train_sent in train_corpus:
			#print(unicodeToAscii(train_sent))
			sent1, sent2, label = train_sent.strip().split('\t')
			
			word_list_1 = sent1.split()
			replace_index_1 = np.random.rand(len(word_list_1)) < word_dropout_rate # will return list of ture / false. ex) [True False True]
			replaced_word_list_1 = []
			for word, replace in zip(word_list_1, replace_index_1):
				if replace:
					word = 'unk'
				replaced_word_list_1.append(word)
			sent1=' '.join(replaced_word_list_1)
			#print(unicodeToAscii(sent1))

			word_list_2 = sent2.split()
			replace_index_2 = np.random.rand(len(word_list_2)) < word_dropout_rate # will return list of ture / false. ex) [True False True]
			replaced_word_list_2 = []
			for word, replace in zip(word_list_2, replace_index_2):
				if replace:
					word = 'unk'
				replaced_word_list_2.append(word)
			sent2=' '.join(replaced_word_list_2)
			#print(unicodeToAscii(sent2))

			w_augmented.write(sent1.strip()+' EndOfSentence\t'+sent2.strip()+' EndOfSentence\t'+ label.strip() +'\n')
