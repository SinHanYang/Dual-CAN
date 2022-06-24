# %%
import os
import json
import socket
import urllib3
import requests
import pandas as pd
from pathlib import Path
import random
import tagme
import time
tagme.GCUBE_TOKEN ="" # put yor tagme password at here
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')


# %%
'''
Dict:
    id: directory name
    text: news content
    desc_list: a list with entity descriptions 
'''
def get_news(path):
    data=[]
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    count=0
    total_desc=0
    for Dict in data:
        entity_desc_list=[]
        try:
            # get entity_desc from wiki
            if(pd.isna(Dict['text'])):
                Dict['text']=""
                Dict['desc_list']=entity_desc_list
                continue
            annotations=tagme.annotate(Dict['text'])
            if(annotations==None):
                Dict['desc_list']=entity_desc_list
                continue
            entitylist=[]
                # Print annotations with a score higher than 0.1
            for ann in annotations.get_annotations(0.3):
                A,B,score=str(ann).split(" -> ")[0],str(ann).split(" -> ")[1].split(" (score: ")[0],str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
                entitylist.append(B)

            wiki_list=[]
            for entity in entitylist:
                page_py = wiki_wiki.page(entity)
                wiki_list.append(page_py.title)
            wiki_list=list(set(wiki_list))
            for name in wiki_list:
                page_py=wiki_wiki.page(name)
                    #print("name:",page_py.title)
                try: 
                    entity_desc_list.append(page_py.summary)
                except json.decoder.JSONDecodeError:
                    print("small,JSONERROR")
                    continue
                except TimeoutError:
                    print("small,TimeoutError")
                    continue
                except socket.timeout:
                    print("small,socker.timeout")
                    continue
                except urllib3.exceptions.ReadTimeoutError:
                    print("small,urllib3.exceptions.ReadTimeoutError")
                    continue
                except requests.exceptions.ReadTimeout:
                    print("small,requests.exceptions.ReadTimeout")
                    continue
        except (TimeoutError,requests.exceptions.ReadTimeout,urllib3.exceptions.ReadTimeoutError,socket.timeout) as e:
            Dict['desc_list']=entity_desc_list
            print("big,TimeoutError")
            continue
        Dict['desc_list']=entity_desc_list
        print("len:",len(Dict['desc_list']))
        total_desc+=len(Dict['desc_list'])
        count+=1
        print("epoch:",i," | count:",count,"|total:",len(data))
    
    return (total_desc/len(data)),data

SPLIT=['train','eval','test']
DATA=['train.json','dev.json','test.json']
def get_data(type,data_path):
    num,total=get_news(data_path)
    print("avg:",num)
    with open(f"coaid_data/{SPLIT[type]}.json","w") as f:
        for Dict in total:
            json.dump(Dict,f)
            f.write('\n')


for i in range(3):
    get_data(i,DATA[i])
