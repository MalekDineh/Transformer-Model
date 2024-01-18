from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from datasets import load_from_disk

cleaned_data = load_from_disk("data/wmt17_de-en_cleaned.hf")

class BPE_Tokenizer:
    def __init__(self, vocab_size):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[BOS]", "[EOS]"])
        self.tokenizer.train_from_iterator(cleaned_data, trainer)
