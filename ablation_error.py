import torch
from src.models import DecoderModel
from src.utils import generate_synthetic_data, normalize_data, tokenize_normalized
from src.train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x, y = generate_synthetic_data()
x, y, _, _ = normalize_data(x.numpy(), y.numpy())
x, y = torch.tensor(x), torch.tensor(y)

# Error correction with repetition
repeats = [1, 3, 5]
for r in repeats:
    model = DecoderModel(input_dim=1, vocab_size=10, seq_length=4 * r)
    targets = torch.stack([tokenize_normalized(yi, length=4) for yi in y]).repeat(1, r)
    train_model(model, x, y, tokenization="normalized", device=device)
    
    # Majority voting
    model.eval()
    with torch.no_grad():
        tokens = model(x)
        token_chunks = tokens.view(-1, r, 4)
        voted_tokens = torch.mode(token_chunks, dim=1)[0]
        pred = torch.tensor([detokenize_normalized(t) for t in voted_tokens], dtype=torch.float32).view(-1, 1)
    mse = mean_squared_error(y.numpy(), pred.numpy())
    print(f"Repeats {r}: MSE = {mse:.4f}")
