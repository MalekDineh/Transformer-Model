import re, collections
import string
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
def format_word(word):
    return ' '.join(list(word))


# vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
# 'n e w e s t </w>':6, 'w i d e s t </w>':3}
text = "Machine learning helps in understanding complex patterns. Learning machine languages can be complex yet rewarding. Natural language processing unlocks valuable insights from data. Processing language naturally is a valuable skill in machine learning. Understanding natural language is crucial in machine learning."

text_processed = text.lower().translate(str.maketrans('', '', string.punctuation))

words = text_processed.split()
print(words)

words_freq = dict(collections.Counter(words))

vocab = {format_word(word): count for word, count in words_freq.items()}

num_merges = 64
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    # print(best)
