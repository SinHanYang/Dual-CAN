import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import RobertaModel,AutoModel

class Classifier(nn.Module):
    def __init__(self,hidden_size:int):
        super(Classifier, self).__init__()
        self.l1=nn.Linear(hidden_size,2) 

    def forward(self,input):
        ans=self.l1(input)
        y=F.softmax(ans,dim=1)
        return y

class Model(nn.Module):
    def __init__(self,bert_size):
        super(Model, self).__init__()
        self.bert_model = AutoModel.from_pretrained("michiyasunaga/LinkBERT-base")
        self.classifier=Classifier(bert_size)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,input_ids,mask):
        _,o2= self.bert_model(input_ids,attention_mask=mask,return_dict=False)

        y=self.classifier(o2)
        return y

    def cal_loss(self,pred,target):
        return  self.loss(pred,target)
