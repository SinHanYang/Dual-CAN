from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch: List[List[List[str]]], max_sent_len: int = None, max_sent_num: int = None
    ) -> List[List[List[int]]]:
        # change [news[sent[str,str,...]],] to [news[sent[int,int,...]],]
        padded_list=[]
        for batch_tokens in batch:
            # get each sentence into fix size
            batch_ids = [self.encode(tokens) for tokens in batch_tokens]
            to_len = max(len(ids) for ids in batch_ids) if max_sent_len is None else max_sent_len
            padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
            padded_list.append(padded_ids)
        
        pad_list=[self.pad_id]*max_sent_len
        padded_list=pad_to_num(padded_list,max_sent_num,pad_list)
        return padded_list


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

def pad_to_num(seqs: List[List[List[int]]], to_len: int, padding: List) -> List[List[List[int]]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds
