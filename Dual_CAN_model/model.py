import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F

def split(a, n): #split in to n part
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

class WordEncoder(nn.Module):
    def __init__(self, embeddings: torch.tensor, hidden_size: int, num_layers: int, dropout: float,max_len: int,max_sent_num:int):
        super(WordEncoder, self).__init__()

        self.L=max_len
        self.sent_num=max_sent_num
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.embed_dim=300
        self.gru=nn.GRU(input_size=self.embed_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout,bidirectional=True)
        self.h_proj=nn.Linear(hidden_size*2,hidden_size)
        self.utw=nn.Linear(hidden_size,1,bias=False)
        
    def forward(self,batch):
        # change[[sent_list[[],[]]],[sent_list[[],[]]]] to [[vi],[vi],...]
        # Ns:max sentence number
        #batch=sum(batch) [batch,Ns,sent_max_len]->[batch*N,max_len]
        input=self.embed(batch)
        hit,_ =self.gru(input) #dim=[batch*Ns,sent_max_len,hidden_size*2] , if last dim == nn.linear's input it's ok 
        uit=torch.tanh(self.h_proj(hit)) # u = tanh( Wh+b ) #[batch*Ns,L,h]
        ait=torch.exp(self.utw(uit))# ait=exp(uit* utw) #[batch*Ns,L,1]

        #ait=exp(uit*utw)/sum(exp(uit*utw))
        aitsum=torch.sum(ait,dim=2,keepdim=True)#[batch*Ns,1]
        aitsum.type(torch.float)
        aitsum=aitsum.expand(-1,self.L,-1) #[batch*Ns,L,1]
        ait=torch.div(ait,aitsum)#[batch*Ns,L,1]

        #vi=sum_{t=1}ait*hit
        ait=torch.transpose(ait,1,2)
        vi=torch.bmm(ait,hit) #[batch*Ns,1,2h]
        batch_size=int(vi.shape[0]/self.sent_num)
        vi=list(split(vi,batch_size)) #[batch,Ns,2h]
        vi=torch.stack(vi)
        vi=torch.squeeze(vi,2)
        return vi

class SentEncoder(nn.Module):
    def __init__(self,hidden_size: int, num_layers: int, dropout: float):
        super(SentEncoder, self).__init__()
        self.gru=nn.GRU(input_size=hidden_size*2,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout,bidirectional=True)
    
    def forward(self,batch):
        x,_ =self.gru(batch) # xdim: [batch_size,N,hiddensize*2] N: sentence in a news
        return x

class CoAttention(nn.Module):
    def __init__(self,hidde_size:int,attention_size:int): #hidden_size:d, attention_size:k
        super(CoAttention, self).__init__()

        self.hidden_size=hidde_size
        self.Wl=nn.Parameter(torch.zeros(size=(hidde_size*2,hidde_size*2)),requires_grad=True)
        self.Ws=nn.Parameter(torch.zeros(size=(attention_size,hidde_size*2)),requires_grad=True)
        self.Wc=nn.Parameter(torch.zeros(size=(attention_size,hidde_size*2)),requires_grad=True)
        self.whs=nn.Parameter(torch.zeros(size=(1,attention_size)),requires_grad=True)
        self.whc=nn.Parameter(torch.zeros(size=(1,attention_size)),requires_grad=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.Wl.data.uniform_(-1.0,1.0)
        self.Ws.data.uniform_(-1.0,1.0)
        self.Wc.data.uniform_(-1.0,1.0)
        self.whs.data.uniform_(-1.0,1.0)
        self.whc.data.uniform_(-1.0,1.0)

    def forward(self,new_batch,entity_desc_batch):
        # news_batch: [batch size, N, hidden size *2] hidden size:h
        S=torch.transpose(new_batch,1,2)
        # entity_desc_batch: [batch size, T, hidden size *2] T: entity description sentences for a news
        C=torch.transpose(entity_desc_batch,1,2)

        attF=torch.tanh(torch.bmm(torch.transpose(C,1,2),torch.matmul(self.Wl,S))) #dim [batch_size,T,N]

        WsS=torch.matmul(self.Ws,S) #dim[batch,a,N] a:attention size
        WsC=torch.matmul(self.Wc,C) #dim[batch,a,T]

        Hs=torch.tanh(WsS+torch.bmm(WsC,attF)) #dim[batch,a,N]
        Hc=torch.tanh(WsC+torch.bmm(WsS,torch.transpose(attF,1,2))) #dim[batch,a,T]


        a_s=F.softmax(torch.matmul(self.whs,Hs),dim=2) #dim[batch,1,N]
        a_c=F.softmax(torch.matmul(self.whc,Hc),dim=2) #dim[batch,1,T]

        s=torch.bmm(a_s,new_batch) # dim[batch,1,2h]
        c=torch.bmm(a_c,entity_desc_batch)#[batch,1,2h]
        return s,c,a_s,a_c
    
class Classifier(nn.Module):
    def __init__(self,hidden_size:int):
        super(Classifier, self).__init__()
        self.l1=nn.Linear(hidden_size*8,hidden_size*4) # news sent+kb sent concatenation
        self.l2=nn.Linear(hidden_size*4,2) 

    def forward(self,news_batch,entity_desc_batch):
        total=torch.cat((news_batch,entity_desc_batch),2)
        layer1=self.l1(total)
        ans=self.l2(layer1)
        y=torch.squeeze(ans,1)
        #y=F.softmax(ans,dim=1)
        return y

class Model(nn.Module):
    def __init__(self, embeddings: torch.tensor, hidden_size: int, word_num_layers: int, dropout: float, max_len: int,sent_num_layers:int,attention_size:int,max_sent_num:int,max_desc_sent_num,max_tweet_sent_num):
        super(Model, self).__init__()
        self.news_word_encoder=WordEncoder(embeddings,hidden_size,word_num_layers,dropout,max_len,max_sent_num)
        self.desc_word_encoder=WordEncoder(embeddings,hidden_size,word_num_layers,dropout,max_len,max_desc_sent_num)
        self.tweet_word_encoder=WordEncoder(embeddings,hidden_size,word_num_layers,dropout,max_len,max_tweet_sent_num)
        self.sent_encoder=SentEncoder(hidden_size,sent_num_layers,dropout)
        self.co_attention=CoAttention(hidden_size,attention_size)
        self.classifier=Classifier(hidden_size)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,news_batch,entity_desc_batch,tweet_batch):
        news_word_encoding=self.news_word_encoder(news_batch)
        contents=self.sent_encoder(news_word_encoding)
        desc_word_encoding=self.desc_word_encoder(entity_desc_batch)
        #description=self.sent_encoder(desc_word_encoding)
        
        tweet_word_encoding=self.tweet_word_encoder(tweet_batch)
        #tweet=self.sent_encoder(tweet_word_encoding)

        contents_att, desc_att, contents_cd_att_weight,desc_att_weight=self.co_attention(contents,desc_word_encoding)
        content_desc=torch.cat((contents_att,desc_att),2)
        content_desc=F.softmax(content_desc,dim=2) #dim[batch,1,4h]

        contents_att2,tweet_att, contents_ct_att_weight,tweet_att_weight=self.co_attention(contents,tweet_word_encoding)
        content_tweet=torch.cat((contents_att2,tweet_att),2)
        content_tweet=F.softmax(content_tweet,dim=2) #dim[batch,1,4h]

        y=self.classifier(content_desc,content_tweet)
        return y,contents_cd_att_weight,desc_att_weight,contents_ct_att_weight,tweet_att_weight

    def cal_loss(self,pred,target):
        return  self.loss(pred,target)
