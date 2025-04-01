import torch
import numpy as np
from src.models import DecoderModel, MixtureDensityModel
from src.utils import generate_synthetic_data, normalize_data
from src.train import train_model, evaluate_model
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate bimodal data
x, y1 = generate_synthetic_data(func="sin")
_, y2 = generate_synthetic_data(func="exp")
y = torch.cat([y1[:500], y2[500:]], dim=0)
x, y, _, _ = normalize_data(x.numpy(), y.numpy())
x, y = torch.tensor(x), torch.tensor(y)

# Models
decoder = DecoderModel(input_dim=1, vocab_size=10, seq_length=4)
mdn = MixtureDensityModel(input_dim=1, mixtures=5)

# Train
train_model(decoder, x, y, tokenization="normalized", device=device)
train_model(mdn, x, y, device=device)

# Sample for density estimation
decoder.eval()
with torch.no_grad():
    tokens = decoder(x)
    samples = [detokenize_normalized(t) for t in tokens]
    pi, mu, sigma = mdn(x)
    mdn_samples = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(pi),
        torch.distributions.Normal(mu, sigma)
    ).sample((1000,)).mean(dim=0)

# Plot
plt.hist(y.numpy(), bins=50, alpha=0.5, label="True", density=True)
plt.hist(samples, bins=50, alpha=0.5, label="Decoder", density=True)
plt.hist(mdn_samples.numpy(), bins=50, alpha=0.5, label="MDN", density=True)
plt.legend()
plt.savefig("results/density/density_est.png")
plt.show()
