import torch
from src.models import PointwiseModel, HistogramModel, DecoderModel
from src.utils import generate_synthetic_data, normalize_data
from src.train import train_model, evaluate_model
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate data
x, y = generate_synthetic_data(func="sin")
x, y, _, y_scaler = normalize_data(x.numpy(), y.numpy())
x, y = torch.tensor(x), torch.tensor(y)

# Models
pointwise = PointwiseModel(input_dim=1)
histogram = HistogramModel(input_dim=1, bins=16)
decoder_norm = DecoderModel(input_dim=1, vocab_size=10, seq_length=4)
decoder_unnorm = DecoderModel(input_dim=1, vocab_size=10, seq_length=7)  # E=1, M=4 + 2 for sign

# Train
train_model(pointwise, x, y, device=device)
train_model(histogram, x, y, device=device)
train_model(decoder_norm, x, y, tokenization="normalized", device=device)
train_model(decoder_unnorm, x, y, tokenization="unnormalized", device=device)

# Evaluate
mse_p, pred_p = evaluate_model(pointwise, x, y, device=device)
mse_h, pred_h = evaluate_model(histogram, x, y, device=device)
mse_dn, pred_dn = evaluate_model(decoder_norm, x, y, tokenization="normalized", device=device)
mse_du, pred_du = evaluate_model(decoder_unnorm, x, y, tokenization="unnormalized", device=device)

print(f"Pointwise MSE: {mse_p:.4f}")
print(f"Histogram MSE: {mse_h:.4f}")
print(f"Decoder (Norm) MSE: {mse_dn:.4f}")
print(f"Decoder (Unnorm) MSE: {mse_du:.4f}")

# Plot
plt.scatter(x, y, label="True", alpha=0.5)
plt.scatter(x, pred_p, label="Pointwise", alpha=0.5)
plt.scatter(x, pred_h, label="Histogram", alpha=0.5)
plt.scatter(x, pred_dn, label="Decoder (Norm)", alpha=0.5)
plt.scatter(x, pred_du, label="Decoder (Unnorm)", alpha=0.5)
plt.legend()
plt.savefig("results/synthetic/curve_fitting.png")
plt.show()
