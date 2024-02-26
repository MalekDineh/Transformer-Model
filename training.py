import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from dataset import load_dataset_l
from modelling.Transformer import TransformerModel
from modelling.LRS import TransformerScheduler, get_optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from modelling.Tokenizer import MyGPT2Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using {device} for training.")

# Initialize the transformer model
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
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params} trainable parameters.')

# Initialize the dataloader
train_ds, val_ds, test_ds = load_dataset_l()
train_indices = list(range(8000))
val_indices = list(range(500))
train_subset = Subset(train_ds, train_indices)
val_subset = Subset(val_ds, val_indices)
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

# Initialize the loss function, optimizer, and learning rate scheduler
loss_fn = CrossEntropyLoss(ignore_index=0)
optimizer = get_optimizer(model)
scheduler = TransformerScheduler(optimizer, d_model=256, warmup_steps=4000)

train_losses = []
val_losses = []

en_tokenizer = MyGPT2Tokenizer(prefix="en")
de_tokenizer = MyGPT2Tokenizer(prefix="de")

# Training loop
for epoch in range(30):
    # Training
    model.train()
    total_train_loss = 0
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
    for batch in train_bar:
        optimizer.zero_grad()
        tgt, tgt_mask, src, src_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        tgt_input, tgt_mask_input = tgt[:, :-1], tgt_mask[:, :-1]
        tgt_output = tgt[:, 1:]
        outputs = model(src, tgt_input, src_mask, tgt_mask_input)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        train_bar.set_postfix({'batch_train_loss': loss.item()})
    train_losses.append(total_train_loss/len(train_loader))

    torch.save(model.state_dict(), f'models/transformer/model_epoch_{epoch+1}.pth')

    # Validation
    model.eval()
    total_val_loss = 0
    val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}')
    with torch.no_grad():
        for batch in val_bar:
            tgt, tgt_mask, src, src_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            tgt_input, tgt_mask_input = tgt[:, :-1], tgt_mask[:, :-1]
            tgt_output = tgt[:, 1:]
            outputs = model(src, tgt_input, src_mask, tgt_mask_input)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
            total_val_loss += loss.item()
            train_bar.set_postfix({'batch_train_loss': loss.item()})
    val_losses.append(total_val_loss/len(val_loader))

    print(f"Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_loader)}, Val Loss: {total_val_loss/len(val_loader)}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
