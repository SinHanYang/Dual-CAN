import torch
import os
import json
import torch.optim as optim
from tqdm import trange
from sklearn.metrics import classification_report

def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

class Trainer:
    def __init__(self,args,model,tr_set,tr_size,dev_set,dev_size):
        self.args = args
        self.model=model
        self.tr_set=tr_set
        self.tr_size=tr_size
        self.dev_set=dev_set
        self.dev_size=dev_size

    def train(self):
        for param in self.model.bert_model_news.parameters():
            param.requires_grad = False
        for param in self.model.bert_model_entity.parameters():
            param.requires_grad = False
        for param in self.model.bert_model_tweet.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)

        epoch_pbar = trange(self.args.num_epoch, desc="Epoch")
        best_acc=0
        for epoch in epoch_pbar:
            # Training loop - iterate over train dataloader and update model weights
            self.model.train()
            total_loss,train_acc=0,0
            for batches in self.tr_set:
                optimizer.zero_grad()
                input_ids_news=batches["input_ids_news"].to(self.args.device)
                masks_news=batches["masks_news"].to(self.args.device)

                input_ids_entity=batches["input_ids_entity"].to(self.args.device)
                masks_entity=batches["masks_entity"].to(self.args.device)

                input_ids_tweet=batches["input_ids_tweet"].to(self.args.device)
                masks_tweet=batches["masks_tweet"].to(self.args.device)
                
                y=batches["label_list"].to(self.args.device)
                pred=self.model(input_ids_news,masks_news,input_ids_entity,masks_entity,input_ids_tweet,masks_tweet)
                _, label= torch.max(pred,1)
                y=y.to(torch.long)
                loss = self.model.cal_loss(pred, y)  # compute loss
                loss.backward()                 # compute gradient (backpropagation)
                optimizer.step()                    # update model with optimizer
                correct = evaluation(label, y) # 計算此時模型的 training accuracy 
                train_acc += correct
                total_loss += loss.item()
                #small_train_info_json = {"epoch": epoch,"count":count,"one_batch_train_Acc":correct/self.args.batch_size*100}
                #print(f"{'#' * 30} insize_epoch: {str(small_train_info_json)} {'#' * 30}")
                #with open(os.path.join(self.args.output_dir, "log.txt"), mode="a") as fout:
                #    fout.write(json.dumps(small_train_info_json) + "\n")

            train_info_json = {"epoch": epoch,"train_loss": total_loss/self.tr_size,"train_Acc":train_acc/self.tr_size}
            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")
            with open(os.path.join(self.args.output_dir, "log.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")

            self.model.eval()                                # set model to evalutation mode
            total_acc=0
            ans_list=[]
            preds=[]
            with torch.no_grad():
                for batches in self.dev_set:                       
                    input_ids_news=batches["input_ids_news"].to(self.args.device)
                    masks_news=batches["masks_news"].to(self.args.device)

                    input_ids_entity=batches["input_ids_entity"].to(self.args.device)
                    masks_entity=batches["masks_entity"].to(self.args.device)

                    input_ids_tweet=batches["input_ids_tweet"].to(self.args.device)
                    masks_tweet=batches["masks_tweet"].to(self.args.device)
                    y=batches["label_list"].to(self.args.device)
                    for ans_label in y:
                        ans_label=int(ans_label)
                        ans_list.append(ans_label)
                    y=y.to(self.args.device)
                    pred=self.model(input_ids_news,masks_news,input_ids_entity,masks_entity,input_ids_tweet,masks_tweet)
                    _, label= torch.max(pred,1)
                    y=y.to(torch.long)
                    correct=evaluation(label,y)
                    total_acc+=correct
                    for p in label.cpu().numpy():
                        preds.append(p)
            
            print(classification_report(ans_list, preds))
            with open(os.path.join(self.args.output_dir, "log.txt"), mode="a") as f:
                f.write(classification_report(ans_list,preds))
                                                                                                                                                   
            valid_info_json = {"epoch": epoch,"val_Acc":total_acc/self.dev_size*100}
            print(f"{'#' * 30} VALID: {str(valid_info_json)} {'#' * 30}")
            with open(os.path.join(self.args.output_dir, "log.txt"), mode="a") as fout:
                fout.write(json.dumps(valid_info_json) + "\n")

            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(self.model, "{}/ckpt.model".format(self.args.ckpt_dir))
                print('saving model with acc {:.3f}\n'.format(total_acc/self.dev_size*100))
                with open(os.path.join(self.args.output_dir, "best_valid_log.txt"), mode="a") as fout:
                    fout.write(json.dumps(valid_info_json) + "\n")
