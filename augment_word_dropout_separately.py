import sys
import numpy as np

import unicodedata
import string

# 윈도우에서 스패인어 보기 위해.
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


word_dropout_rate = 0.1 # 0.2

#output = 'WD_%0.2f_aug_train_twitter.txt'%word_dropout_rate
output = 'WD_%0.2f_aug_train_text.txt'%word_dropout_rate

number_of_augmented_align = 2
number_of_augmented_none = 8

with open(output, 'w', encoding ='utf-8') as w_augmented,\
	open('./'+sys.argv[1], 'r', encoding='utf-8') as align,\
	open('./'+sys.argv[2], 'r', encoding='utf-8') as none:
	align_corpus = align.readlines()
	none_corpus = none.readlines()

	# 본래 align_data 
	for align_sent in align_corpus:
		w_augmented.write(align_sent)
		
	# number_of_augmented_align - 1 만큼 augmented 한 데이터를 추가함.
	for _ in range(number_of_augmented_align - 1):
		for align_sent in align_corpus:
			#print(unicodeToAscii(train_sent))
			sent1, sent2, label = align_sent.strip().split('\t')
			
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

			w_augmented.write(sent1.strip()+'\t'+sent2.strip()+'\t'+ label.strip() +'\n')


	# 본래 train_data 
	for none_sent in none_corpus:
		w_augmented.write(none_sent)

	# number_of_augmented_none - 1 만큼 augmented 한 데이터를 추가함.
	for _ in range(number_of_augmented_none - 1):
		for none_sent in none_corpus:
			#print(unicodeToAscii(train_sent))
			sent1, sent2, label = none_sent.strip().split('\t')
			
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

			w_augmented.write(sent1.strip()+'\t'+sent2.strip()+'\t'+ label.strip() +'\n')
