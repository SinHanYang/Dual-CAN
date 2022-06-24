# %%
import pandas as pd
import json
import numpy as np

# %%
news=[]

# %%
real=pd.read_csv('./CoAID/NewsRealCOVID-19.csv')
for index, row in real.iterrows():
    Dict={}
    Dict['label']='real'
    Dict['id']=str('real')+str(row['id'])
    if(pd.isna(real.iloc[index,6])==False):
        Dict['text']=row['content']
    else:
        tmp=""
        if(pd.isna(real.iloc[index,4])==False):
            tmp+=row['title']
        if(pd.isna(real.iloc[index,5])==False):
            tmp+=row['newstitle']
        Dict['text']=tmp
    Dict['tweet_list']=[]
    news.append(Dict)

# %%
fake=pd.read_csv('./CoAID/NewsFakeCOVID-19.csv')
for index, row in fake.iterrows():
    Dict={}
    Dict['label']='fake'
    Dict['id']='fake'+str(row['id'])
    #print(row['content'],type(row['content']))
    if(pd.isna(fake.iloc[index,11])==False):
        Dict['text']=row['content']
    else:
        tmp=""
        if(pd.isna(fake.iloc[index,9])==False):
            tmp+=row['title']
        if(pd.isna(fake.iloc[index,10])==False):
            tmp+=row['newstitle']
        Dict['text']=tmp
    Dict['tweet_list']=[]
    if(row['type']=='article'):
        news.append(Dict)

# %%
total_count=0

# %%
real_tweet=pd.read_csv('./CoAID/NewsRealCOVID-19_tweets_expanded.csv')
for index,row in real_tweet.iterrows():
    if(row['full_text'] is not np.nan):
        for Dict in news:
            if(Dict['id']==('real'+str(row['index']))):
                Dict['tweet_list'].append(row['full_text'])
                total_count+=1


# %%
fake_tweet=pd.read_csv('./CoAID/NewsFakeCOVID-19_tweets_expanded.csv')
for index,row in fake_tweet.iterrows():
    if(row['full_text'] is not np.nan):
        for Dict in news:
            if(Dict['id']==('fake'+str(row['index']))):
                Dict['tweet_list'].append(row['full_text'])
                total_count+=1


# %%
from sklearn.model_selection import train_test_split

# %%
train,test=train_test_split(news,random_state=777,train_size=0.8)
train,val=train_test_split(train,random_state=777,train_size=0.9)

# %%
with open("covid_data_article/train.json","w") as f:
    for Dict in train:
        json.dump(Dict,f)
        f.write('\n')


with open("covid_data_article/dev.json","w") as f:   
    for Dict in val:
        json.dump(Dict,f)
        f.write('\n')

with open("covid_data_article/test.json","w") as f:
    for Dict in test:
        json.dump(Dict,f)
        f.write('\n')


