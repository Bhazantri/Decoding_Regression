
#### `src/models.py`
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class PointwiseHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super(PointwiseHead, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Enforce [0,1] output

class HistogramHead(nn.Module):
    def __init__(self, hidden_dim=128, bins=16):
        super(HistogramHead, self).__init__()
        self.fc = nn.Linear(hidden_dim, bins)
        self.bins = bins
    
    def forward(self, x):
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(0, 1, self.bins).to(x.device)
        return (probs * bin_centers).sum(dim=-1, keepdim=True)

class DecoderHead(nn.Module):
    def __init__(self, hidden_dim=128, vocab_size=10, seq_length=4, num_layers=3, nhead=4):
        super(DecoderHead, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead=nhead, dim_feedforward=512), num_layers
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        device = x.device
        if targets is None:  # Inference
            tokens = torch.zeros(batch_size, self.seq_length, dtype=torch.long).to(device)
            for t in range(self.seq_length):
                emb = self.embedding(tokens[:, :t+1]) + self.pos_embedding[:, :t+1]
                out = self.transformer(emb.transpose(0, 1), x.unsqueeze(0)).transpose(0, 1)
                logits = self.fc(out[:, -1, :])
                tokens[:, t] = torch.argmax(logits, dim=-1)
            return tokens
        else:  # Training
            emb = self.embedding(targets) + self.pos_embedding
            out = self.transformer(emb.transpose(0, 1), x.unsqueeze(0)).transpose(0, 1)
            return self.fc(out)

class MixtureDensityHead(nn.Module):
    def __init__(self, hidden_dim=128, mixtures=5):
        super(MixtureDensityHead, self).__init__()
        self.mixtures = mixtures
        self.pi = nn.Linear(hidden_dim, mixtures)  # Mixture weights
        self.mu = nn.Linear(hidden_dim, mixtures)  # Means
        self.sigma = nn.Linear(hidden_dim, mixtures)  # Standard deviations
    
    def forward(self, x):
        pi = torch.softmax(self.pi(x), dim=-1)
        mu = self.mu(x)
        sigma = torch.nn.functional.elu(self.sigma(x)) + 1  # Ensure positivity
        return pi, mu, sigma

class PointwiseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(PointwiseModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.head = PointwiseHead(hidden_dim)
    
    def forward(self, x):
        enc = self.encoder(x)
        return self.head(enc)

class HistogramModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, bins=16):
        super(HistogramModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.head = HistogramHead(hidden_dim, bins)
    
    def forward(self, x):
        enc = self.encoder(x)
        return self.head(enc)

class DecoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, vocab_size=10, seq_length=4, nhead=4):
        super(DecoderModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.head = DecoderHead(hidden_dim, vocab_size, seq_length, num_layers, nhead)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
    
    def forward(self, x, targets=None):
        enc = self.encoder(x)
        return self.head(enc, targets)

class MixtureDensityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, mixtures=5):
        super(MixtureDensityModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.head = MixtureDensityHead(hidden_dim, mixtures)
    
    def forward(self, x):
        enc = self.encoder(x)
        return self.head(enc)
