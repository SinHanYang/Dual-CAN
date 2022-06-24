import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F

class WordEncoder(nn.Module):
    def __init__(self, embeddings: torch.tensor, hidden_size: int, num_layers: int, dropout: float,max_len: int,max_sent_num:int):
        super(WordEncoder, self).__init__()

        self.L=max_len
        self.sent_num=max_sent_num
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.embed_dim=100
        self.gru=nn.GRU(input_size=self.embed_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout,bidirectional=True)
        
    def forward(self,batch):
        #[batch,N*sent_max_len,h]
        input=self.embed(batch)
        hit,_ =self.gru(input) #dim=[batch,N*sent_max_len,hidden_size*2] , if last dim == nn.linear's input it's ok 
        x=hit[:,-1,:]
        return x
    
class Classifier(nn.Module):
    def __init__(self,hidden_size:int):
        super(Classifier, self).__init__()
        self.l1=nn.Linear(hidden_size*6,hidden_size*4) # news sent+kb sent concatenation
        self.l2=nn.Linear(hidden_size*4,2) 

    def forward(self,news_batch,entity_desc_batch,tweet_batch):
        total=torch.cat((news_batch,entity_desc_batch,tweet_batch),1)
        layer1=self.l1(total)
        y=self.l2(layer1)
        #y=F.softmax(ans,dim=1)
        return y

class Model(nn.Module):
    def __init__(self, embeddings: torch.tensor, hidden_size: int, word_num_layers: int, dropout: float, max_len: int,sent_num_layers:int,attention_size:int,max_sent_num:int,max_desc_sent_num,max_tweet_sent_num):
        super(Model, self).__init__()
        self.news_word_encoder=WordEncoder(embeddings,hidden_size,word_num_layers,dropout,max_len,max_sent_num)
        self.desc_word_encoder=WordEncoder(embeddings,hidden_size,word_num_layers,dropout,max_len,max_desc_sent_num)
        self.tweet_word_encoder=WordEncoder(embeddings,hidden_size,word_num_layers,dropout,max_len,max_tweet_sent_num)
        self.classifier=Classifier(hidden_size)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,news_batch,entity_desc_batch,tweet_batch):
        news_word_encoding=self.news_word_encoder(news_batch)
        desc_word_encoding=self.desc_word_encoder(entity_desc_batch)
        #description=self.sent_encoder(desc_word_encoding)
        
        tweet_word_encoding=self.tweet_word_encoder(tweet_batch)
        #tweet=self.sent_encoder(tweet_word_encoding)

        y=self.classifier(news_word_encoding,desc_word_encoding,tweet_word_encoding)
        return y

    def cal_loss(self,pred,target):
        return  self.loss(pred,target)
