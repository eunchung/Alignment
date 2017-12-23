import os
import re
import tweepy
import time
from tweepy import OAuthHandler


class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = '0k3jgF7srhKancNh7fNSTP8ht'
        consumer_secret = 'Z3Md4W49Mk85b6vSXJG9rYROKfGbPyhpDPCoBO0eS8w6kn81s2'
        access_token = '420794457-ZKqlX6OQ7F9aNZpyNT0ofIgwi7aljMx4XET2nqDj'
        access_token_secret = 'IdeKjPqMWYdoMhCy8Sg10laZR3em4auQAucXdy9u4BI1M'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
    
    def get_tweets(self, query, count = 10, RT_number = 0):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
 
        try:
            filtered_fetched_tweets = []
            filtered_fetched_tweet_ids = []
            # call twitter api to fetch tweets. count 만큼 lang="es" 서치하고, 그 중에서 리트윗횟수 높은거만 추출하자.
            fetched_tweets = self.api.search(q = query, count = count, lang="es") 
            #그냥 id_str 이 아니라, retweeted_status.id_str 을 써야, 제대로 된 트위터 status 아이디로 감.
            
            # 리트윗 수 높은 것만 추출.
            for tweet in fetched_tweets:
                if tweet.retweet_count >= RT_number and tweet.text: #not in filtered_fetched_tweets:
                    filtered_fetched_tweets.append(tweet.text)
                    filtered_fetched_tweet_ids.append(tweet.retweeted_status.id_str)
                    
            
            # parsing tweets one by one
            for tweet, id in zip(filtered_fetched_tweets, filtered_fetched_tweet_ids) :
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
 
                # saving text of tweet
                parsed_tweet['raw_text'] = tweet.strip()
                parsed_tweet['tweet_id'] = id  
                tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            print("Error : " + str(e))
            
    
    def get_replies(self, query, tweet_id, count):
        '''
        트위터의 reply 들을 얻기위한 코드
        참조 : https://stackoverflow.com/questions/29928638/getting-tweet-replies-to-a-particular-tweet-from-a-particular-user
        
        그냥 tweet.id_str 이 아니라, tweet.retweeted_status.id_str 을 써야, 제대로 된 트위터 status 아이디로 감. !!
        '''
        replies = []
        
        try:
            filtered_fetched_replies = []
            fetched_tweets = self.api.search(q = query, since_id = tweet_id, count = count, lang="es") 
            
            # 그냥 리트윗은 제외하고, 리트윗에 맨션 있는것만 추출.
            for tweet in fetched_tweets:
                if tweet.in_reply_to_status_id==int(tweet_id):
                    filtered_fetched_replies.append(tweet.text)

            # parsing tweets one by one
            for reply in filtered_fetched_replies:
                parsed_reply = {}
                parsed_reply['raw_text'] = reply.strip()    
                replies.append(parsed_reply)
  
            # return parsed tweets
            return replies
 
        except tweepy.TweepError as e:
            print("Error : " + str(e))
 
def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    count = 100
    query = 'RT' #independencia'
    RT_number = 100
    
    if not os.path.exists('./data/tweet/crawled'):
        os.makedirs('./data/tweet/crawled')
    
    print(query)
    
    count_file = 0
    for _ in range(100):
        count_file +=1
        if count_file % 5 == 0:
            print('crawled %s times' % count_file)

        time_now = time.time()
        with open('./data/tweet/crawled/' + query + '_'+ str(RT_number)+'_'+str(time_now)+'.txt', 'w',encoding='UTF-8') as w:
            # calling function to get tweets
            w.write('RT 로 리트윗 검색. 리트윗 100개 넘은 것들 + 그것에 달린 reply 들 기록. \n')
            tweets = api.get_tweets(query, count, RT_number)

            for tweet in tweets:
                query_reply = tweet['raw_text'].split()[1][:-1]
                replies = api.get_replies(query_reply, tweet['tweet_id'],100)

                if replies:
                    w.write(tweet['raw_text']+'\n')
                    for reply in replies:
                        w.write(reply['raw_text']+'\n')
                    w.write('\n')
            
            # wait 3*60 sec to avoid 'Rate limit exceeded'
            time.sleep(3*60)

            
if __name__ == "__main__":
    # calling main function
    main()