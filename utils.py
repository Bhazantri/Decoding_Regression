import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalized Tokenization (Section 3.2)
def tokenize_normalized(y, base=10, length=4, y_min=None, y_max=None):
    if y_min is None or y_max is None:
        y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min)  # Normalize to [0,1]
    tokens = []
    for _ in range(length):
        digit = int(y_scaled * base)
        tokens.append(min(digit, base - 1))  # Cap at vocab_size - 1
        y_scaled = (y_scaled * base) - digit
    return torch.tensor(tokens, dtype=torch.long)

def detokenize_normalized(tokens, base=10, length=4, y_min=0, y_max=1):
    value = 0
    for i, t in enumerate(tokens):
        value += t * (base ** -(i + 1))
    return value * (y_max - y_min) + y_min

# Unnormalized Tokenization (Section 3.2)
def tokenize_unnormalized(y, base=10, E=1, M=4):
    y = y.item() if isinstance(y, torch.Tensor) and y.numel() == 1 else y
    sign = 1 if y >= 0 else -1
    y = abs(y)
    if y == 0:
        return torch.zeros(E + M + 2, dtype=torch.long)  # Sign, Sign_e, E, M
    exponent = int(np.floor(np.log(y) / np.log(base))) if y > 0 else 0
    mantissa = y / (base ** exponent)
    exp_tokens = [int(d) for d in f"{abs(exponent):0{E}d}"]
    mantissa_tokens = []
    for _ in range(M):
        digit = int(mantissa * base)
        mantissa_tokens.append(min(digit, base - 1))
        mantissa = (mantissa * base) - digit
    return torch.tensor([sign + 1, 1 if exponent >= 0 else 0] + exp_tokens + mantissa_tokens, dtype=torch.long)

def detokenize_unnormalized(tokens, base=10, E=1, M=4):
    sign = -1 if tokens[0] == 0 else 1
    exp_sign = -1 if tokens[1] == 0 else 1
    exponent = exp_sign * int("".join(map(str, tokens[2:2+E])))
    mantissa = 0
    for i, t in enumerate(tokens[2+E:]):
        mantissa += t * (base ** -(i + 1))
    return sign * (base ** exponent) * mantissa

# Hamming Distance-based Tokenization (Appendix A.3)
def tokenize_hamming(y, length=3):
    y = (y - y.min()) / (y.max() - y.min())  # Normalize to [0,1]
    n = 2 ** length
    mapping = [(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,1), (1,0,1), (0,1,1), (1,1,0)]
    idx = int(y * (n - 1))
    return torch.tensor(mapping[idx % len(mapping)], dtype=torch.long)

def detokenize_hamming(tokens, length=3):
    mapping = [(0,0,0), (0,0,1), (0,1,0), (1,0,0), (1,1,1), (1,0,1), (0,1,1), (1,1,0)]
    idx = mapping.index(tuple(tokens.tolist()))
    return idx / (len(mapping) - 1)

# Data normalization
def normalize_data(x, y):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x = torch.tensor(x_scaler.fit_transform(x), dtype=torch.float32)
    y = torch.tensor(y_scaler.fit_transform(y.reshape(-1, 1)), dtype=torch.float32)
    return x, y, x_scaler, y_scaler

# Synthetic data generation (Section 4.1)
def generate_synthetic_data(n_samples=1000, func="sin"):
    x = np.linspace(-5, 5, n_samples)
    if func == "sin":
        y = np.sin(x) + np.random.normal(0, 0.1, n_samples)
    elif func == "exp":
        y = np.exp(-x**2) + np.random.normal(0, 0.1, n_samples)
    else:
        y = x**2 + np.random.normal(0, 0.1, n_samples)
    return torch.tensor(x, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)
