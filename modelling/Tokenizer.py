from transformers import AutoTokenizer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from datasets import load_from_disk

def get_training_corpus(dataset, language='de', batch_size = 1000):

    training_corpus = (dataset['train'][i:i+batch_size][language]
                       for i in range(0, len(dataset['train'][language]), batch_size))
    return training_corpus

def train_tokenizer(dataset, language, vocab_size, outdir):
    training_corpus = get_training_corpus(dataset, language)
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token='<|pad|>', eos_token='<|eos|>', bos_token='<|bos|>')
    trained_tokenizer = base_tokenizer.train_new_from_iterator(training_corpus, vocab_size)
    trained_tokenizer.save_pretrained(outdir)

if __name__ == '__main__':
    cleaned_data = load_from_disk("data/wmt17_de-en_cleaned.hf")
    train_tokenizer(cleaned_data, 'de', 50000, "models/tokenizer_de")