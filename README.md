# Dual-CAN
Code for EACL 2023 Findings "Entity-Aware Dual Co-Attention Network for Fake News Detection"

## How to Run
1. Put data in the same directory (bigru, BERT_based, Dual_CAN_model) of the code.
2. For Bi-GRU and Dual-CAN model, need to preprocess the data by running
```
python preprocess.py --data_dir data
```
3. train and test the model
```
python main.py --data_dir data --output_dir output
python main.py --data_dir data --output_dir output --mode 1
```

## How to Create data
* Follow FakeNewsNet (https://github.com/KaiDMML/FakeNewsNet) and CoAID (https://github.com/cuilimeng/CoAID) instructions to download the data.
* The id we use for the experiments are in `data` directory.
    * For GossipCop, the data_id is the name of the directory. We only use tweet without retweet.
    * For CoAID, we use the first version (05-01-2020). The dataid is the (label+id). Again, We only use tweet without retweet.

### Preprocessing Data of  GossipCop
* Execute the code in `data/GossipCop`
```
python preprocess.py
python preprocess_tweets.py
```
### Preprocessing Data of  CoAID
* Execute the code in `data/CoAID`
```
python get_data.py
python preprocess.py
```

## Dependencies
```
python 3.7
torch 1.7.1
tensorflow 2.4.1
pytorch-lightning 1.2.3
spacy 3.0.5
seqeval 1.2.2
tqdm
numpy
pandas
scikit_learn
tagme
wikipediaapi
```
