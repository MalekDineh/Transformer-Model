import datasets
import re
import tqdm
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
import torch
from modelling.Tokenizer import MyGPT2Tokenizer

class TranslationDataset(Dataset):
    def __init__(self, text, de_tokenizer, en_tokenizer, max_length=74):
        """The dataset class for the translation task.

        Args:
            text (list): A list of dictionaries with the keys "de" and "en".
            de_tokenizer (type): German tokenizer
            en_tokenizer (type): English tokenizer
            max_length (int, optional): Max sequence length. Defaults to 64.

        Returns:
            german_tokens: Tensor of shape (batch_size, max_length)
            english_tokens: Tensor of shape (batch_size, max_length)
            german_attention_mask: Tensor of shape (batch_size, max_length)
            english_attention_mask: Tensor of shape (batch_size, max_length)
        """
        self.text = text
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer
        self.pad_token_id_de = de_tokenizer.tokenizer.pad_token_id
        self.pad_token_id_en = en_tokenizer.tokenizer.pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        de_sentence, en_sentence = self.text[idx]["de"], self.text[idx]["en"]

        # Tokenize the sentences
        de_tokens = self.de_tokenizer.tokenizer.encode(de_sentence)
        en_tokens = self.en_tokenizer.tokenizer.encode(en_sentence)

        # pad with BOS and EOS tokens
        de_tokens = [self.de_tokenizer.tokenizer.bos_token_id] + de_tokens + [self.de_tokenizer.tokenizer.eos_token_id]
        en_tokens = [self.en_tokenizer.tokenizer.bos_token_id] + en_tokens + [self.en_tokenizer.tokenizer.eos_token_id]

        # Pad or truncate the sequences to the specified max_length
        de_tokens = de_tokens + [self.pad_token_id_de] * (self.max_length - len(de_tokens))
        en_tokens = en_tokens + [self.pad_token_id_en] * (self.max_length - len(en_tokens))

        # calculate the attention mask
        # 1 where de_tokens != pad_token_id_de, 0 otherwise
        de_attention_mask = [1 if token != self.pad_token_id_de else 0 for token in de_tokens]
        en_attention_mask = [1 if token != self.pad_token_id_en else 0 for token in en_tokens]

        # assert that all shapes are equal
        assert len(de_tokens) == len(en_tokens) == len(de_attention_mask) == len(en_attention_mask) == self.max_length, f"Shape mismatch: de_tokens: {len(de_tokens)}, en_tokens: {len(en_tokens)}, de_attention_mask: {len(de_attention_mask)}, en_attention_mask: {len(en_attention_mask)}, max_length: {self.max_length}"

        return [torch.tensor(de_tokens), torch.tensor(de_attention_mask), torch.tensor(en_tokens), torch.tensor(en_attention_mask)]

def load_dataset_l():
    en_tokenizer = MyGPT2Tokenizer(prefix="en")
    de_tokenizer = MyGPT2Tokenizer(prefix="de")
    ds = load_from_disk('data/wmt17_de-en_cleaned.hf')

    train_ds = TranslationDataset(ds['train'], de_tokenizer, en_tokenizer)
    val_ds = TranslationDataset(ds['validation'], de_tokenizer, en_tokenizer)
    test_ds = TranslationDataset(ds['test'], de_tokenizer, en_tokenizer)

    return train_ds, val_ds, test_ds

def clean_dataset(dataset, min_length=5, max_length=64, max_ratio=1.5):
    dataset = dataset.copy()
    whitelist = set(
        "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüßABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥")

    def clean_text(text):
        # Remove non-UTF8 characters
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # Remove URLs and HTML tags
        text = re.sub(r"http\S+|www.\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r"<.*?>", "", text)

        # Remove characters not in the whitelist
        text = ''.join(c for c in text if c in whitelist)

        return text

    cleaned_dataset = datasets.DatasetDict()
    for split in dataset.keys():
        data_split = {
            'en': [],
            'de': []
        }

        for data in tqdm.tqdm(dataset[split], desc = split):
            src_text = data["translation"]["en"]
            tgt_text = data["translation"]["de"]

            # Clean source and target texts
            src_text = clean_text(src_text)
            tgt_text = clean_text(tgt_text)

            # Check if the lengths are within the specified range
            if min_length <= len(src_text) <= max_length and min_length <= len(tgt_text) <= max_length:
                # Check the ratio between source and target lengths
                ratio = len(src_text) / len(tgt_text)
                if 1/max_ratio <= ratio <= max_ratio:
                    data_split['en'].append(src_text)
                    data_split['de'].append(tgt_text)

        cleaned_dataset[split] = datasets.Dataset.from_dict(data_split)
    return cleaned_dataset

if __name__ == '__main__':
    ds = load_dataset("wmt17", "de-en")
    cleaned_dataset = clean_dataset(ds)
    cleaned_dataset.save_to_disk("data/wmt17_de-en_cleaned.hf")

