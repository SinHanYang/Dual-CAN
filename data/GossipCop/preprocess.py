# %%
import os
import json
import socket
import urllib3
import requests
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
def get_news(rootdir,start,end,label):
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
        entity_desc_list=[]
        try:
            # get entity_desc from wiki
            annotations=tagme.annotate(Dict['text'])
            if(annotations==None):
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
            print("big,TimeoutError")
            continue
        Dict['desc_list']=entity_desc_list
        print("len:",len(Dict['desc_list']))
        total_desc+=len(Dict['desc_list'])
        lists.append(Dict)
        count+=1
        print("epoch:",i," | label:",label," | count:",count)
    
    return total_desc,lists

def get_data(i):
    rootdir='./fakenewsnet_dataset/gossipcop/fake'
    tota,lista=get_news(rootdir,100*i,100*(i+1),'fake')
    rootdir='./fakenewsnet_dataset/gossipcop/real'
    totb,listb=get_news(rootdir,150*i,150*(i+1),'real')
    lists=lista+listb
    random.shuffle(lists)
    train_size=int(len(lists)*0.75*0.9)
    valid_size=train_size+int(len(lists)*0.075)
    print("train_size:",train_size)
    print("valid_size",valid_size-train_size)
    print("test_size",len(lists)-valid_size)
    print("average_desc_number:",(tota+totb)/len(lists))

    with open("data/train.json","a") as f:
        for Dict in lists[0:train_size]:
            json.dump(Dict,f)
            f.write('\n')


    with open("data/dev.json","a") as f:   
        for Dict in lists[train_size:valid_size]:
            json.dump(Dict,f)
            f.write('\n')

    with open("data/test.json","a") as f:
        for Dict in lists[valid_size:]:
            json.dump(Dict,f)
            f.write('\n')

for i in range(7,40):
    get_data(i)
