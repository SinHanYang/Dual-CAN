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
            input_ids_news=batches["input_ids_news"].to(self.args.device)
            masks_news=batches["masks_news"].to(self.args.device)

            input_ids_entity=batches["input_ids_entity"].to(self.args.device)
            masks_entity=batches["masks_entity"].to(self.args.device)

            input_ids_tweet=batches["input_ids_tweet"].to(self.args.device)
            masks_tweet=batches["masks_tweet"].to(self.args.device)
            #token_type_ids=batches["token_type_ids"].to(self.args.device)
            ans=batches["label_list"]
            ids=batches["id_list"]
            for ans_label in ans:
                ans_list.append(ans_label)
            for index in ids:
                id_list.append(index)
            with torch.no_grad():                   # disable gradient calculation
                output = self.model(input_ids_news,masks_news,input_ids_entity,masks_entity,input_ids_tweet,masks_tweet)                     # forward pass (compute output)
                prob=output[:,1]
                _, pred= torch.max(output,1)
                for y in pred.cpu().numpy():
                    preds.append(y)
                for p in prob.cpu().numpy():
                    prob_1.append(p)

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

