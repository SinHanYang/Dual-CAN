import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import RobertaModel,AutoModel

class Classifier(nn.Module):
    def __init__(self,hidden_size:int):
        super(Classifier, self).__init__()
        self.l1=nn.Linear(hidden_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,2) 

    def forward(self,input):
        l1=self.l1(input)
        relu=self.relu(l1)
        ans=self.l2(relu)
        y=F.softmax(ans,dim=1)
        return y

class Model(nn.Module):
    def __init__(self,bert_size):
        super(Model, self).__init__()
        self.bert_model_news = AutoModel.from_pretrained("michiyasunaga/LinkBERT-base")
        self.bert_model_entity = AutoModel.from_pretrained("michiyasunaga/LinkBERT-base")
        self.bert_model_tweet = AutoModel.from_pretrained("michiyasunaga/LinkBERT-base")
        self.classifier=Classifier(bert_size*3)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,input_ids_news,mask_news,input_ids_entity,mask_entity,input_ids_tweet,mask_tweet):
        _,o2_news= self.bert_model_news(input_ids_news,attention_mask=mask_news,return_dict=False)
        _,o2_entity= self.bert_model_entity(input_ids_entity,attention_mask=mask_entity,return_dict=False)
        _,o2_tweet= self.bert_model_tweet(input_ids_tweet,attention_mask=mask_tweet,return_dict=False)
        
        o2=torch.cat([o2_news,o2_entity,o2_tweet],dim=1)
        y=self.classifier(o2)
        return y

    def cal_loss(self,pred,target):
        return  self.loss(pred,target)
