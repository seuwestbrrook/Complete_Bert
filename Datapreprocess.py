from torch.utils.data import DataLoader, Dataset
import torch
import random
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from typing import List
# This file is to prepare the data for pretraining BERT.
@dataclass
class PretrainDataSample:
    """
    A data sample for pretraining BERT.
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor
    is_next: torch.Tensor


class PretrainDataset(Dataset):
    """
    Prepare the data for pretraining Bert. It converts raw sentences into the format required for pretraining.
    The dataset generate data for the following two tasks:

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    Each sample is in the following format (see Figure 2):
        [CLS] + masked_sentence_A + [SEP] + masked_sentence_B + [SEP]
    """

    def __init__(
        self,
        data_source,
        tokenizer,
    ):
        self.tokenizer = tokenizer
        self.data_source = data_source

    def __len__(self):
        # the last sentence does not have a next sentence
        return len(self.data_source) - 1

    def mask_sentence(self, token_ids: List[int]):
        masked_lm_label = []

        for i, token_id in enumerate(token_ids):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    # 80% randomly change token to mask token
                    token_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10% randomly change token to a random token
                    token_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                else:
                    # 10% stay the same
                    pass

                masked_lm_label.append(token_id)

            else:
                masked_lm_label.append(self.tokenizer.pad_token_id)

        return token_ids, masked_lm_label

    def get_nsp_pair(self, index):
        """
        when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual
        next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from
        the corpus (labeled as NotNext).
        """
        sentence_a = self.data_source[index]
        is_next = random.random() > 0.5
        if is_next:
            sentence_b = self.data_source[index + 1]
        else:
            sentence_b = self.data_source[random.randint(0, len(self.data_source) - 1)]

        # we suppose that the input is a dictionary with a key "text"
        # otherwise, check _load_* or dataset preparation
        assert isinstance(sentence_a, dict) and isinstance(sentence_b, dict)
        assert "text" in sentence_a and "text" in sentence_b
        return sentence_a["text"], sentence_b["text"], is_next

    def __getitem__(self, index):
        sentence_a, sentence_b, is_next = self.get_nsp_pair(index)

        # 先分词
        token_ids_a = self.tokenizer.encode(sentence_a, add_special_tokens=False)
        token_ids_b = self.tokenizer.encode(sentence_b, add_special_tokens=False)

        # 然后做 masking
        masked_sentence_a, sentence_a_labels = self.mask_sentence(token_ids_a)
        masked_sentence_b, sentence_b_labels = self.mask_sentence(token_ids_b)

        # 拼接输入
        input_ids = (
            [self.tokenizer.cls_token_id]
            + masked_sentence_a
            + [self.tokenizer.sep_token_id]
            + masked_sentence_b
            + [self.tokenizer.sep_token_id]
        )

        labels = (
            [self.tokenizer.pad_token_id]
            + sentence_a_labels
            + [self.tokenizer.pad_token_id]
            + sentence_b_labels
            + [self.tokenizer.pad_token_id]
        )

        segment_label = [0] * (1 + len(masked_sentence_a) + 1) + [1] * (len(masked_sentence_b) + 1)

        return PretrainDataSample(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(segment_label, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            is_next=torch.tensor(is_next, dtype=torch.long),
        )



def collate_fn(batch: List[PretrainDataSample], pad_token_id: int):
    input_ids = pad_sequence(
        [item.input_ids for item in batch], padding_value=pad_token_id, batch_first=True
    )
    token_type_ids = pad_sequence(
        [item.token_type_ids for item in batch],
        padding_value=0,
        batch_first=True,
    )
    labels = pad_sequence(
        [item.labels for item in batch], padding_value=pad_token_id, batch_first=True
    )
    is_next = torch.cat([item.is_next.unsqueeze(0) for item in batch])

    return PretrainDataSample(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        labels=labels,
        is_next=is_next,
    )
