import sys
import numpy as np
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


def main():
	output = 'preprocessed_' + sys.argv[1]
	
	count = 0
	with open(output, 'w', encoding ='utf-8') as w, open('./'+sys.argv[1], 'r', encoding='utf-8') as tweet:


		tweet_corpus = tweet.readlines()
		for i, line in enumerate(tweet_corpus):
			sent1, sent2 = line.strip().split('\t')
			sent1 = clean_tweet(sent1)
			sent2 = clean_tweet(sent2)
			if (len(sent1) == 0 or len(sent2) == 0):
				print('[twitter_crawled_data] %d-th sentence is empty after cleaning' % (i+1))
				continue
			word_list_1 = toktok.tokenize(' '.join(nltk.wordpunct_tokenize(sent1)))
			word_list_2 = toktok.tokenize(' '.join(nltk.wordpunct_tokenize(sent2)))
			sent1=' '.join(word_list_1)
			sent2=' '.join(word_list_2)
			w.write(sent1.strip()+' endofsentence\t'+sent2.strip()+' endofsentence\ttemp_label\n')

			count += 1
			if count % 100 == 0:
				print(count)


if __name__ == "__main__":
	main()