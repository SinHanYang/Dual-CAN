import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import os
import copy
import torch

from dataset import KB_CoDataset
from model import Model
from utils import Vocab
from train import Trainer
from evaluate import Tester
from torch.utils.data import DataLoader

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]

def json_reader(path):
    data=[]
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    label_idx_path = args.cache_dir / "label2idx.json"
    label2idx: Dict[str, int] = json.loads(label_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json_reader(path) for split, path in data_paths.items()}
    datasets: Dict[str, KB_CoDataset] = {
        split: KB_CoDataset(split_data, vocab, label2idx, args.max_sent_len,args.max_sent_num,args.max_desc_sent_num,args.max_single_desc,args.max_single_tweet,args.max_tweet_sent_num,mode=split)
        for split, split_data in data.items()
    }
    for split,split_dataset in datasets.items():
        if(split=="train" and args.mode==0):
            tr_size=len(split_dataset)
            print("tr_size:",tr_size)
            num_classes=split_dataset.num_classes
            tr_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last=False,
                num_workers=0, pin_memory=False)
        elif(split=="eval" and args.mode==0):
            dev_size=len(split_dataset)
            print("dev_size:",dev_size)
            dv_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last=False,
                num_workers=0, pin_memory=False)
        elif(args.mode==1):
            test_size=len(split_dataset)
            print("test_size:",test_size)
            test_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last=False,
                num_workers=0, pin_memory=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = Model(
        embeddings,
        args.hidden_size, 
        args.word_num_layers, 
        args.dropout,
        args.max_sent_len,
        args.sent_num_layers,
        args.attention_size,
        args.max_sent_num,
        args.max_desc_sent_num,
        args.max_tweet_sent_num
    )
    model.to(args.device)
    ifexist=os.path.exists(args.output_dir)
    if not ifexist:
        os.makedirs(args.output_dir)
    if args.mode==0: #train/dev
        args_dict_tmp = vars(args)
        args_dict = copy.deepcopy(args_dict_tmp)
        with open(os.path.join(args.output_dir, "param.txt"), mode="w") as f:
            f.write("============ parameters ============\n")
            print("============ parameters =============")
            for k, v in args_dict.items():
                f.write("{}: {}\n".format(k, v))
                print("{}: {}".format(k, v))
        trainer=Trainer(args,model,tr_set,tr_size,dv_set,dev_size)
        trainer.train()
    else: #test
        tester=Tester(args,model,test_set,test_size,datasets["test"])
        tester.test()
    
    



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./tri_data/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/ckpt.model",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./output/tri_5_3_2022/",
    )

    # data
    parser.add_argument("--max_sent_len", type=int, default=120)
    parser.add_argument("--max_single_desc", type=int, default=4)
    parser.add_argument("--max_single_tweet", type=int, default=1)
    parser.add_argument("--max_sent_num", type=int, default=50)
    parser.add_argument("--max_desc_sent_num", type=int, default=100)
    parser.add_argument("--max_tweet_sent_num", type=int, default=100)

    # model
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--attention_size", type=int, default=300)
    parser.add_argument("--word_num_layers", type=int, default=2)
    parser.add_argument("--sent_num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:1"
    )
    parser.add_argument("--num_epoch", type=int, default=20)
    #
    parser.add_argument("--mode", type=int, help="train:0, test:1", default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
