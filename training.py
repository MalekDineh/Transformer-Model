import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from dataset import load_dataset_l
from modelling.Transformer import TransformerModel
from modelling.LRS import TransformerScheduler, get_optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using {device} for training.")

# Initialize the transformer model
vocab_size = 50000  # replace with your actual vocab size
d_model = 128  # replace with your actual d_model
n_heads = 4  # replace with your actual n_heads
num_encoder_layers = 4  # replace with your actual num_encoder_layers
num_decoder_layers = 4  # replace with your actual num_decoder_layers
dim_feedforward = 64  # replace with your actual dim_feedforward
dropout = 0.2  # replace with your actual dropout
max_len = 64  # replace with your actual max_len

model = TransformerModel(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len, device)
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params} trainable parameters.')

# Initialize the dataloader
train_ds, val_ds, test_ds = load_dataset_l()
train_indices = list(range(7000))
val_indices = list(range(500))
train_subset = Subset(train_ds, train_indices)
val_subset = Subset(val_ds, val_indices)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

# Initialize the loss function, optimizer, and learning rate scheduler
loss_fn = CrossEntropyLoss()
optimizer = get_optimizer(model)
scheduler = TransformerScheduler(optimizer, d_model=128, warmup_steps=4000)

train_losses = []
val_losses = []

# Training loop
for epoch in range(20):
    # Training
    model.train()
    total_train_loss = 0
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
    for batch in train_bar:
        optimizer.zero_grad()
        tgt, tgt_mask, src, src_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        outputs = model(src, tgt, src_mask, tgt_mask)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        train_bar.set_postfix({'batch_train_loss': loss.item()})
    train_losses.append(total_train_loss/len(train_loader))

    # Validation
    model.eval()
    total_val_loss = 0
    val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}')
    with torch.no_grad():
        for batch in val_bar:
            tgt, tgt_mask, src, src_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            outputs = model(src, tgt, src_mask, tgt_mask)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), tgt.view(-1))
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
