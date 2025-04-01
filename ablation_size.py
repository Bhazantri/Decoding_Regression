import torch
from src.models import DecoderModel
from src.utils import generate_synthetic_data, normalize_data
from src.train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x, y = generate_synthetic_data()
x, y, _, _ = normalize_data(x.numpy(), y.numpy())
x, y = torch.tensor(x), torch.tensor(y)

# Vary decoder size
configs = [
    {"num_layers": 1, "nhead": 1, "hidden_dim": 32},
    {"num_layers": 3, "nhead": 4, "hidden_dim": 128},
    {"num_layers": 5, "nhead": 8, "hidden_dim": 256}
]

for config in configs:
    model = DecoderModel(input_dim=1, **config, vocab_size=10, seq_length=4)
    train_model(model, x, y, tokenization="normalized", device=device)
    mse, _ = evaluate_model(model, x, y, tokenization="normalized", device=device)
    print(f"Config {config}: MSE = {mse:.4f}")
