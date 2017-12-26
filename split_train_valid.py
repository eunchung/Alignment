import sys
import numpy as np
from sklearn.model_selection import train_test_split

output_train = 'train_' + sys.argv[1]
output_valid = 'valid_' + sys.argv[1]
#output_test = 'test_twitter.txt'
#output_test = 'test_text.txt'
#output_test = 'test_all.txt' # 이건 위의 두 데이터를 합쳐서 만듬.


import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
toktok = ToktokTokenizer()

import unicodedata
import string
import math

# 윈도우에서 스패인어 보기 위해.
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9_:]+)|(#[A-Za-z0-9]+)|(RT)|(\w+:\/\/\S+)|([^A-Za-z0-9áÁéÉíÍóÓúÚñÑüÜªº€¡!?¿(){}<>,.…@#$%&‘’“”« »「 」\[\]~\:\;\-\_\+*\'\`\"\^])", " ", tweet).split()) # 최종버전


with open(output_train, 'w', encoding ='utf-8') as w_train,\
	open(output_valid, 'w', encoding ='utf-8') as w_valid,\
	open(output_test, 'w', encoding ='utf-8') as w_test,\
	open('./'+sys.argv[1], 'r', encoding='utf-8') as data, \
	open('./'+sys.argv[2], 'r', encoding='utf-8') as test_data:

	corpus = data.readlines()
	clean_tokenized_corpus = []
	for i, line in enumerate(corpus):
		#print(unicodeToAscii(line))
		sent1, sent2, label = line.strip().split('\t')
		sent1 = clean_tweet(sent1)
		sent2 = clean_tweet(sent2)
		if (len(sent1) == 0 or len(sent2) == 0):
			print('[train_data] %d-th sentence is empty after cleaning' % (i+1))
			continue
		word_list_1 = toktok.tokenize(' '.join(nltk.wordpunct_tokenize(sent1)))
		word_list_2 = toktok.tokenize(' '.join(nltk.wordpunct_tokenize(sent2)))
		sent1=' '.join(word_list_1)
		sent2=' '.join(word_list_2)
		clean_tokenized_corpus.append(sent1.strip()+'\t'+sent2.strip()+'\t'+ label.strip() +'\n')


	train, valid = train_test_split(clean_tokenized_corpus, test_size=0.1, random_state=6)
	
	print('length of train_data', len(train))
	print('length of valid_data', len(valid))
	
	for train_sent in train:
		sent1, sent2, label = train_sent.strip().split('\t')
		w_train.write(sent1.strip()+' endofsentence\t'+sent2.strip()+' endofsentence\t'+ label.strip() +'\n')
		#w_train.write(sent1.strip()+'\t'+sent2.strip()+'\t'+ label.strip() +'\n') # 나중에 augmented 할때 endofsentence 추가하는 방식을 사용. 이때 추가해두면 worddropout 시에 unk 되버림.


	for valid_sent in valid:
		sent1, sent2, label = valid_sent.strip().split('\t')
		w_valid.write(sent1.strip()+' endofsentence\t'+sent2.strip()+' endofsentence\t'+ label.strip() +'\n')
		

	test_corpus = test_data.readlines()
	for i, line in enumerate(test_corpus):
		sent1, sent2, label = line.strip().split('\t')
		sent1 = clean_tweet(sent1)
		sent2 = clean_tweet(sent2)
		if (len(sent1) == 0 or len(sent2) == 0):
			print('[test_data] %d-th sentence is empty after cleaning' % (i+1))
			continue
		word_list_1 = toktok.tokenize(' '.join(nltk.wordpunct_tokenize(sent1)))
		word_list_2 = toktok.tokenize(' '.join(nltk.wordpunct_tokenize(sent2)))
		sent1=' '.join(word_list_1)
		sent2=' '.join(word_list_2)
		w_test.write(sent1.strip()+' endofsentence\t'+sent2.strip()+' endofsentence\t'+ label.strip() +'\n')

