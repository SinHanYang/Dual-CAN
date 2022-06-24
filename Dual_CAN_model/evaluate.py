import torch
import csv
import json
import os
from sklearn.metrics import classification_report,auc,precision_recall_curve

class Tester:
    def __init__(self,args,model,test_set,test_size,dataset):
        self.args = args
        self.model=model
        self.test_set=test_set
        self.test_size=test_size
        self.dataset=dataset

    def test(self):
        self.model = torch.load(self.args.ckpt_path)
        self.model.eval()
        preds = []
        prob_1=[]
        ans_list=[]
        id_list=[]
        att_weight_list=[]
        for batches in self.test_set:                            # iterate through the dataloader
            text=batches["texts"]
            desc=batches["descs"]
            tweet=batches["tweets"]
            ans=batches["label_list"]
            ids=batches["id_list"]
            text,desc,tweet=text.to(self.args.device),desc.to(self.args.device),tweet.to(self.args.device)
            for ans_label in ans:
                ans_list.append(ans_label)
            for index in ids:
                id_list.append(index)
            with torch.no_grad():                   # disable gradient calculation
                output,cd_weight,desc_weight,ct_weight,tweet_weight = self.model(text,desc,tweet)                     # forward pass (compute output)
                prob=output[:,1]
                _, pred= torch.max(output,1)
                for y in pred.cpu().numpy():
                    preds.append(y)
                for p in prob.cpu().numpy():
                    prob_1.append(p)
                for id,y,cd,d,ct,t in zip(ids,pred.cpu().numpy(),cd_weight.cpu().numpy(),desc_weight.cpu().numpy(),ct_weight.cpu().numpy(),tweet_weight.cpu().numpy()):
                    Dict={}
                    Dict['id']=id
                    cd=cd.tolist()
                    Dict['label']=self.dataset.idx2label(y)
                    Dict['content_desc_weight']=cd
                    d=d.tolist()
                    Dict['desc_weight']=d
                    ct=ct.tolist()
                    Dict['content_tweet_weight']=ct
                    t=t.tolist()
                    Dict['tweet_weight']=t
                    att_weight_list.append(Dict)

        print(classification_report(ans_list, preds,digits=4))
         
        #PR-AUC for CoAID
        precision, recall, thresholds = precision_recall_curve(ans_list, prob_1)
        auc_precision_recall = auc(recall, precision)
        print("PR AUC,",auc_precision_recall)

        with open(os.path.join(self.args.output_dir, "report.txt"), mode="w") as f:
            f.write("PR AUC:")
            f.write(str(auc_precision_recall))
            f.write('\n')
            f.write(classification_report(ans_list,preds,digits=4))

        with open(os.path.join(self.args.output_dir, "result.txt"), mode="w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'label'])
            for i,p in enumerate(preds):
                writer.writerow([id_list[i],self.dataset.idx2label(p)])
        
        with open(os.path.join(self.args.output_dir, "att_weight.txt"), mode="w") as fa:
            for Dict in att_weight_list:
                json.dump(Dict,fa)
                fa.write('\n')
        
