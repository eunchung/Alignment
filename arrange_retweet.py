import sys
import os

def main():
	output = './data/tweet/total_retweet.txt'
	
	count = 0 # tweet/retweet pair 수 세기
	with open(output, 'w', encoding ='utf-8') as w:

		for root, dirs, files in os.walk("./data/tweet/crawled/", topdown = False):
			for name in files:
				print(os.path.join(root, name))
				with open(os.path.join(root, name), 'r', encoding = 'utf-8') as r:
					sentence_continue = ""
					finished_list = ['empty']
					next_is_Blank = False
					for sentence in r.readlines():
						count += 1
						if count % 100 == 0:
							print(count)

						if sentence.strip() == "":
							next_is_Blank = True
							count -= 1 # 빈칸은 세지않음.
							
						# tweet 사이에 개행이 있는 tweet 는 sentence_continue 로 두고, 개행들 다 합침.
						if sentence.strip() != "" and '@' not in sentence:
							sentence_continue = sentence_continue + sentence.strip()
							count -= 1 # 개행되어 있는 건 하나의 트윗으로 세기때문에 count 하지않음.
							continue
						elif sentence.strip() != "":
							sentence_finished = sentence_continue
							sentence_continue = sentence.strip()
							finished_list.append(sentence_finished)

							if next_is_Blank:
								next_is_Blank = False
								finished_list.append('empty')
								count -= 1 # empty 표시는 세지않음.

						
					next_is_RT = False
					finished_list.pop(1) # 한글 한 줄 빼기
					count -= 1 
					for sentence in finished_list:
					 	if sentence == 'empty':
					 		next_is_RT = True
					 		continue
					 	if next_is_RT == True:
					 		RT = sentence
					 		next_is_RT = False
					 		continue
					 	w.write(RT+'\t'+sentence+'\n')
					count -= 1 # 파일의 마지막 빈 한줄 지움.
			print('Total number of tweet-retweet pair:',count) #미세하게 다를때가 있네..
			

if __name__ == "__main__":
	main()