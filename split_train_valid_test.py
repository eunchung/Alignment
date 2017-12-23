import sys
import numpy as np
from sklearn.model_selection import train_test_split

output_train = 'train_' + sys.argv[1]
output_valid = 'valid_' + sys.argv[1]
output_test = 'test_' + sys.argv[1]

import re

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
	open('./'+sys.argv[1], 'r', encoding='utf-8') as data:

	corpus = data.readlines()
	train, test = train_test_split(corpus, test_size=0.2, random_state=6)
	valid =test[ : math.floor(len(test)/2)]
	test = test[math.floor(len(test)/2) : ]

	print('length of train_data', len(train))
	print('length of valid_data', len(valid))
	print('length of test_data', len(test))

	for train_sent in train:
		#print(unicodeToAscii(train_sent))
		sent1, sent2, label = train_sent.strip().split('\t')
		sent1 = clean_tweet(sent1)
		sent2 = clean_tweet(sent2)
		if (len(sent1) == 0 or len(sent2) == 0):
			continue
		w_train.write(sent1.strip()+' EndOfSentence\t'+sent2.strip()+' EndOfSentence\t'+ label.strip() +'\n')

	for valid_sent in valid:
		sent1, sent2, label = valid_sent.strip().split('\t')
		sent1 = clean_tweet(sent1)
		sent2 = clean_tweet(sent2)
		if (len(sent1) == 0 or len(sent2) == 0):
			continue
		w_valid.write(sent1.strip()+' EndOfSentence\t'+sent2.strip()+' EndOfSentence\t'+ label.strip() +'\n')
		

	for test_sent in test:
		sent1, sent2, label = test_sent.strip().split('\t')
		sent1 = clean_tweet(sent1)
		sent2 = clean_tweet(sent2)
		if (len(sent1) == 0 or len(sent2) == 0):
			continue
		w_test.write(sent1.strip()+' EndOfSentence\t'+sent2.strip()+' EndOfSentence\t'+ label.strip() +'\n')

