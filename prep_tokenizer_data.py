from datasets import load_from_disk
from modelling.Tokenizer import prepare_data_for_gpt_tokenizer

"""
This script is used to prepare the data for the GPT2Tokenizer.
It loads the dataset from the disk and prepares the data for the GPT2Tokenizer.
The vocab and merges files are saved in the data/tokenizer folder.
"""

if __name__ == '__main__':
    ds = load_from_disk('data/wmt17_de-en_cleaned.hf')
    prepare_data_for_gpt_tokenizer(ds['train'])


