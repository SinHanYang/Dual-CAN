# %%
import os
import json
import socket
import urllib3
import requests
from pathlib import Path
import random
import time


# %%
'''
Dict:
    id: directory name
    text: news content
    desc_list: a list with entity descriptions 
'''
def get_news(rootdir,roottweet,start,end,label):
    lists=[]
    count=0
    total_desc=0
    dir_list=os.listdir(rootdir)
    dir_list.sort()
    for dir in dir_list[start:end]:
        Dict={}
        Dict['label']=label
        Dict['id']=dir
        dir=Path(dir)
        path=rootdir/ dir / "news content.json"
        if(os.path.exists(path)==False):
            continue
        content=json.loads(path.read_text())
        Dict['text']=content['text']
        #handle tweets
        tweets_path=roottweet/ dir / "tweets"
        tweets_file=os.listdir(tweets_path)
        tweets_file.sort()
        tweet_list=[]
        for tweet in tweets_file:
            tweet=Path(tweet)
            path=tweets_path/tweet
            if(os.path.exists(path)==False):
                continue
            tweet_content=json.loads(path.read_text())
            tweet_list.append(tweet_content['text'])
        Dict['desc_list']=tweet_list
        print("len:",len(Dict['desc_list']))
        total_desc+=len(Dict['desc_list'])
        lists.append(Dict)
        count+=1
        print("epoch:",i," | label:",label," | count:",count)
    
    return total_desc,lists

def get_data(i):
    rootdir='./fakenewsnet_dataset/gossipcop/fake'
    roottweet='./fakenewsnet_withcomments/gossipcop/fake'
    tota,lista=get_news(rootdir,roottweet,100*i,100*(i+1),'fake')
    rootdir='./fakenewsnet_dataset/gossipcop/real'
    roottweet='./fakenewsnet_withcomments/gossipcop/real'
    totb,listb=get_news(rootdir,roottweet,150*i,150*(i+1),'real')
    lists=lista+listb
    random.shuffle(lists)
    train_size=int(len(lists)*0.75*0.9)
    valid_size=train_size+int(len(lists)*0.075)
    print("train_size:",train_size)
    print("valid_size",valid_size-train_size)
    print("test_size",len(lists)-valid_size)
    print("average_desc_number:",(tota+totb)/len(lists))

    with open("tweets_data/train.json","a") as f:
        for Dict in lists[0:train_size]:
            json.dump(Dict,f)
            f.write('\n')


    with open("tweets_data/dev.json","a") as f:   
        for Dict in lists[train_size:valid_size]:
            json.dump(Dict,f)
            f.write('\n')

    with open("tweets_data/test.json","a") as f:
        for Dict in lists[valid_size:]:
            json.dump(Dict,f)
            f.write('\n')

for i in range(7,40):
    get_data(i)
