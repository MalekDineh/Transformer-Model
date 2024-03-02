import json
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers import trainers
from tokenizers import models

class MyGPT2Tokenizer:
    def __init__(self,prefix="en",add_bos_token=False):
        self.tokenizer = GPT2Tokenizer(vocab_file=f"data/tokenizer/{prefix}_vocab.json", merges_file=f"data/tokenizer/{prefix}_merges.json",unk_token="[UNK]",bos_token="[BOS]",eos_token="[EOS]",pad_token="[PAD]",add_bos_token=add_bos_token)
        
    def encode(self, text):
        if isinstance(text, list):
            return [self.tokenizer.encode(txt.lower()) for txt in text]
        elif isinstance(text, str):
            return self.tokenizer.encode(text.lower())
        else:
            raise TypeError("Input must be of type list or string.")
    
    def decode(self, tokens):
        if isinstance(tokens[0], list):
            return [self.tokenizer.decode(token) for token in tokens]
        elif isinstance(tokens[0], int):
            return self.tokenizer.decode(tokens)
        else:
            raise TypeError("Input must be of type list or int.")

def prepare_data_for_gpt_tokenizer(text_data):
    """Tokenizes the text data and saves the vocab and merges files for the GPT2Tokenizer.

    Args:
        text_data (list): List of dictionaries with the keys "de" and "en".
    """
    de_tokenizer, en_tokenizer = tokenize_data(text_data)

    # save complete tokenizers as json files
    de_tokenizer.save_tokenizer("de")
    en_tokenizer.save_tokenizer("en")

    # read the tokenizers and save the english and german vocab and merges files
    for prefix in ["de", "en"]:
        with open(f"data/tokenizer/{prefix}_tokenizer.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(f"data/tokenizer/{prefix}_vocab.json", "w") as f:
            json.dump(data["model"]["vocab"], f)
        with open(f"data/tokenizer/{prefix}_merges.json", "w") as f:
            json.dump(data["model"]["merges"], f)

def tokenize_data(text_data, max_vocab_length=50_000):
    """Tokenizes the text data and returns the vocabularies and tokenizers.

    Args:
        text_data (list): List of dictionaries with the keys "de" and "en".
        max_vocab_length (int, optional): Number of maximal vocabs. Defaults to 50_000.

    Returns: German and English Vocabulary, German and English Tokenizer
    """
    # split german and english text
    de_data = [element["de"] for element in text_data]
    en_data = [element["en"] for element in text_data]

    # create seperate tokenizers
    de_tokenizer = HuggBPETokenizer(max_vocab_size=max_vocab_length)
    en_tokenizer = HuggBPETokenizer(max_vocab_size=max_vocab_length)

    # train tokenizers
    de_tokenizer.train_on_data(de_data)
    en_tokenizer.train_on_data(en_data)

    return de_tokenizer, en_tokenizer


class HuggBPETokenizer:
    def __init__(self, max_vocab_size=50_000):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],unk_token="[UNK]",vocab_size=max_vocab_size)

    def train_on_data(self, data):
        self.tokenizer.train_from_iterator(data, trainer=self.trainer)

    def save_tokenizer(self, prefix: str):
        self.tokenizer.save(f"data/tokenizer/{prefix}_tokenizer.json", pretty=True)

    def load_tokenizer(self, prefix: str):
        self.tokenizer = Tokenizer.from_file(f"data/tokenizer/{prefix}_tokenizer.json")

    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
