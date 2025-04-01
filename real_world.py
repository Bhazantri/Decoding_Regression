import torch
import pandas as pd
from src.models import PointwiseModel, HistogramModel, DecoderModel
from src.utils import normalize_data
from src.train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load UCI dataset (example: Airfoil)
data = pd.read_csv("data/airfoil.csv")  # Placeholder: assumes data is downloaded
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
x, y, _, _ = normalize_data(x, y)
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Models
pointwise = PointwiseModel(input_dim=x.shape[1])
histogram = HistogramModel(input_dim=x.shape[1], bins=64)
decoder = DecoderModel(input_dim=x.shape[1], vocab_size=10, seq_length=4)

# Train
train_model(pointwise, x, y, device=device)
train_model(histogram, x, y, device=device)
train_model(decoder, x, y, tokenization="normalized", device=device)

# Evaluate
mse_p, _ = evaluate_model(pointwise, x, y, device=device)
mse_h, _ = evaluate_model(histogram, x, y, device=device)
mse_d, _ = evaluate_model(decoder, x, y, tokenization="normalized", device=device)

print(f"Pointwise MSE: {mse_p:.4f}")
print(f"Histogram MSE: {mse_h:.4f}")
print(f"Decoder MSE: {mse_d:.4f}")
