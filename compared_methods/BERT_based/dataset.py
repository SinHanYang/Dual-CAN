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
        alltext=""
        for sent in tokenize.sent_tokenize(instance['text']):
            if(news_count>=self.max_sent_num):
                break
            alltext+=str(sent)
            news_count+=1
        
        for desc in instance['desc_list']:
            count=0
            if(desc_count>self.max_desc_sent_num):
                break
            for sent in tokenize.sent_tokenize(desc):
                if(count>=self.max_single_desc or desc_count>=self.max_desc_sent_num):
                    break
                alltext+=str(sent)
                count+=1
                desc_count+=1

        for tweet in instance['tweet_list']:
            count=0
            if(tweet_count>=self.max_tweet_sent_num):
                break
            for sent in tokenize.sent_tokenize(tweet):
                if(count>=self.max_single_tweet or tweet_count>=self.max_tweet_sent_num):
                    break
                alltext+=str(sent)
                count+=1
                tweet_count+=1
        
        inputs = self.tokenizer.encode_plus(
            alltext,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids=torch.tensor(inputs['input_ids'])
        #token_type_id=torch.tensor(inputs['token_type_ids'])
        masks=torch.tensor(inputs['attention_mask'])
        if (self.mode=='test'):
            return  (label,ids,masks,index)
        else:
            return  (label,ids,masks,None)

    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples) -> Dict:   
        batch={}
        label_list=[]
        if(self.mode=="test"):
            batch["id_list"]=[]
            for s in samples:
                batch["id_list"].append(s[3])
        #else:
        for s in samples:
            label_list.append(s[0])
        label_list=torch.tensor(label_list)
        batch["label_list"]=label_list
        ids=[s[1] for s in samples]
        masks=[s[2] for s in samples]
        #token_type_id=[s[3] for s in samples]
        
        ids=pad_sequence(ids,batch_first=True)
        masks=pad_sequence(masks,batch_first=True)
        #token_type_id=pad_sequence(token_type_id,batch_first=True)
    
        ids=torch.tensor(ids)
        masks=torch.tensor(masks)
        #token_type_id=torch.tensor(token_type_id)
        batch['input_ids']=ids
        batch['masks']=masks
        #batch['token_type_ids']=token_type_id
        return batch

            
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

