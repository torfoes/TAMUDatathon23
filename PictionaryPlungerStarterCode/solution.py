# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# Encoder
class Encoder(nn.Module):
    def __init__(self, enc_rnn_size=256, z_size=128):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=2, hidden_size=enc_rnn_size, batch_first=True)
        self.fc_mu = nn.Linear(enc_rnn_size, z_size)
        self.fc_logvar = nn.Linear(enc_rnn_size, z_size)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, dec_rnn_size=512, num_mixture=20, z_size=128):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_size=2 + dec_rnn_size, hidden_size=dec_rnn_size, batch_first=True)
        self.fc = nn.Linear(dec_rnn_size, num_mixture)
        self.fc_z = nn.Linear(z_size, dec_rnn_size)

    def forward(self, x, z):
        z_input = self.fc_z(z).unsqueeze(1)
        x = torch.cat([x, z_input.expand(-1, x.size(1), -1)], dim=-1)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

# SketchRNN Model
class SketchRNN(nn.Module):
    def __init__(self, num_classes=345, enc_rnn_size=256, dec_rnn_size=512, z_size=128, num_mixture=20):
        super(SketchRNN, self).__init__()
        self.encoder = Encoder(enc_rnn_size, z_size)
        self.decoder = Decoder(dec_rnn_size, num_mixture, z_size)
        self.fc = nn.Linear(dec_rnn_size, num_classes)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(x, z)
        out = self.fc(out)
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# For the dataset, since the data files are .npy files, we can create a custom PyTorch Dataset for it
from torch.utils.data import Dataset, DataLoader

class QuickDrawDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_path, self.files[idx]))
        return torch.tensor(data, dtype=torch.float32)

# Training the model
def train_model(model, data_loader, optimizer, num_epochs):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Assuming the batch is a tuple (data, labels)
            data, labels = batch
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Loading the dataset and training the model
dataset = QuickDrawDataset("C:\\Users\\bunin\\Documents\\TAMUDatathon23\\quick_draw_data")
data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

model = SketchRNN(num_classes=len(dataset))
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, data_loader, optimizer, num_epochs=10)  # Adjust num_epochs as needed
