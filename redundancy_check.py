import sys
import numpy as np
from sklearn.model_selection import train_test_split

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

with open('./'+sys.argv[1], 'r', encoding='utf-8') as large_data, \
	open('./'+sys.argv[2], 'r', encoding='utf-8') as small_data:

	check = 0
	large_corpus = large_data.readlines()
	small_corpus = small_data.readlines()
	for i, line in enumerate(small_corpus):
		#print(unicodeToAscii(line))
		if line in large_corpus:
			print((i+1), unicodeToAscii(line))
			check +=1
		else:
			print('-----', (i+1), '-----')

	print('total',check)