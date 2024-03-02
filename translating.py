import torch
from nltk.translate.bleu_score import sentence_bleu
from dataset import load_dataset_l
from torch.utils.data import DataLoader, Subset
from modelling.Transformer import TransformerModel
from tqdm import tqdm
from modelling.Tokenizer import MyGPT2Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

_, _, test_ds = load_dataset_l()
test_indices = list(range(100))
test_subset = Subset(test_ds, test_indices)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

de_tokenizer = MyGPT2Tokenizer(prefix="de")

vocab_size = 50000  
d_model = 256  
n_heads = 4 
num_encoder_layers = 6
num_decoder_layers = 6 
dim_feedforward = 128 
dropout = 0.2 
max_len = 74

model = TransformerModel(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len, device)
model = model.to(device)
model.load_state_dict(torch.load('models/transformer/model_epoch_30.pth'))

def generate_translation(model, test_loader, de_tokenizer, device, max_length=74):
    model.eval()
    test_bar = tqdm(test_loader, desc='Testing')
    translations = []
    with torch.no_grad():
        for batch in test_bar:
            translation = []
            tgt_generated = [de_tokenizer.tokenizer.bos_token_id]
            for _ in range(max_length):
                tgt_padded = tgt_generated + [de_tokenizer.tokenizer.pad_token_id] * (max_length - len(tgt_generated))
                tgt_tensor = torch.LongTensor(tgt_padded).unsqueeze(0).to(device)
                tgt, tgt_mask, src, src_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                output = model(src, tgt_tensor, src_mask, tgt_mask)
                next_token = output.argmax(dim=-1).tolist()[-1][0]
                tgt_generated.append(next_token)
                if next_token == de_tokenizer.tokenizer.eos_token_id:
                    break
            translation.append(tgt_generated)
        translations.append(translation)
    return translations


translations = generate_translation(model, test_loader, de_tokenizer, device)

# Calculate the BLEU score
translations = [de_tokenizer.decode(translation) for translation in translations]

test_subset = [translation[1] for translation in test_subset]

bleu_scores = [sentence_bleu([tgt_sentence], pred_sentence) for tgt_sentence, pred_sentence in zip(test_subset, translations)]
avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f'Average BLEU score: {avg_bleu_score}')