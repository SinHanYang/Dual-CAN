from typing import List, Dict
from nltk import tokenize
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class KB_CoDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        label_mapping: Dict[str, int],
        max_sent_len: int,
        max_sent_num: int,    # max sent_num for a news article
        max_desc_sent_num: int,    # max sent_num for a news KB description
        max_single_desc: int, # max sentence for a single entity description
        max_single_tweet: int,
        max_tweet_sent_num: int,
        mode="train"
    ):
        self.data = data
        self.tokenizer=tokenizer
        self.label_mapping = label_mapping
        self._idx2label = {idx: label for label, idx in self.label_mapping.items()}
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.max_desc_sent_num = max_desc_sent_num
        self.max_single_desc=max_single_desc
        self.max_tweet_sent_num = max_tweet_sent_num
        self.max_single_tweet=max_single_tweet
        self.mode=mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        if(self.mode=="test"):
            index=(instance['id'])
        #else:
        label=self.label2idx(instance['label'])
        #end if else
        news_count=0
        desc_count=0
        tweet_count=0

        # news
        newstext=""
        for sent in tokenize.sent_tokenize(instance['text']):
            if(news_count>=self.max_sent_num):
                break
            newstext+=str(sent)
            news_count+=1
        
        inputs = self.tokenizer.encode_plus(
            newstext,
            padding='max_length',
            truncation=True,
            #pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids_news=torch.tensor(inputs['input_ids'])
        masks_news=torch.tensor(inputs['attention_mask'])
        
        # entity description
        entitytext=""
        for desc in instance['desc_list']:
            count=0
            if(desc_count>self.max_desc_sent_num):
                break
            for sent in tokenize.sent_tokenize(desc):
                if(count>=self.max_single_desc or desc_count>=self.max_desc_sent_num):
                    break
                entitytext+=str(sent)
                count+=1
                desc_count+=1

        inputs = self.tokenizer.encode_plus(
            entitytext,
            padding='max_length',
            truncation=True,
            #pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids_entity=torch.tensor(inputs['input_ids'])
        masks_entity=torch.tensor(inputs['attention_mask'])

        # tweet
        tweettext=""
        for tweet in instance['tweet_list']:
            count=0
            if(tweet_count>=self.max_tweet_sent_num):
                break
            for sent in tokenize.sent_tokenize(tweet):
                if(count>=self.max_single_tweet or tweet_count>=self.max_tweet_sent_num):
                    break
                tweettext+=str(sent)
                count+=1
                tweet_count+=1
        
        inputs = self.tokenizer.encode_plus(
            tweettext,
            padding='max_length',
            truncation=True,
            #pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids_tweet=torch.tensor(inputs['input_ids'])
        masks_tweet=torch.tensor(inputs['attention_mask'])

        if (self.mode=='test'):
            return  (label,ids_news,masks_news,ids_entity,masks_entity,ids_tweet,masks_tweet,index)
        else:
            return  (label,ids_news,masks_news,ids_entity,masks_entity,ids_tweet,masks_tweet,None)

    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples) -> Dict:   
        batch={}
        label_list=[]
        if(self.mode=="test"):
            batch["id_list"]=[]
            for s in samples:
                batch["id_list"].append(s[7])
        #else:
        for s in samples:
            label_list.append(s[0])
        label_list=torch.tensor(label_list)
        batch["label_list"]=label_list

        # ========== news ============== 
        ids=[s[1] for s in samples]
        masks=[s[2] for s in samples]
        
        ids=pad_sequence(ids,batch_first=True)
        masks=pad_sequence(masks,batch_first=True)
    
        ids=torch.tensor(ids)
        masks=torch.tensor(masks)

        batch['input_ids_news']=ids
        batch['masks_news']=masks

        # ========== entity description ============== 
        ids=[s[3] for s in samples]
        masks=[s[4] for s in samples]
        
        ids=pad_sequence(ids,batch_first=True)
        masks=pad_sequence(masks,batch_first=True)
    
        ids=torch.tensor(ids)
        masks=torch.tensor(masks)

        batch['input_ids_entity']=ids
        batch['masks_entity']=masks

        # ========== news ============== 
        ids=[s[5] for s in samples]
        masks=[s[6] for s in samples]
        
        ids=pad_sequence(ids,batch_first=True)
        masks=pad_sequence(masks,batch_first=True)
    
        ids=torch.tensor(ids)
        masks=torch.tensor(masks)

        batch['input_ids_tweet']=ids
        batch['masks_tweet']=masks

        return batch

            
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

