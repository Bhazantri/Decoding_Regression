import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from .utils import tokenize_normalized, detokenize_normalized, tokenize_unnormalized, detokenize_unnormalized

def mdn_loss(pi, mu, sigma, y):
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(y.expand_as(mu))
    weighted_log_prob = log_prob + torch.log(pi)
    return -torch.logsumexp(weighted_log_prob, dim=-1).mean()

def train_model(model, x_train, y_train, tokenization="normalized", epochs=300, lr=0.001, device="cpu", patience=5):
    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if isinstance(model, PointwiseModel) or isinstance(model, HistogramModel):
        criterion = nn.MSELoss()
    elif isinstance(model, MixtureDensityModel):
        criterion = mdn_loss
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    patience_counter = 0
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if isinstance(model, DecoderModel):
            if tokenization == "normalized":
                targets = torch.stack([tokenize_normalized(y) for y in y_train])
            elif tokenization == "unnormalized":
                targets = torch.stack([tokenize_unnormalized(y) for y in y_train])
            logits = model(x_train, targets)
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
        elif isinstance(model, MixtureDensityModel):
            pi, mu, sigma = model(x_train)
            loss = criterion(pi, mu, sigma, y_train)
        else:
            pred = model(x_train)
            loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return losses

def evaluate_model(model, x_test, y_test, tokenization="normalized", device="cpu"):
    model.eval()
    x_test, y_test = x_test.to(device), y_test.to(device)
    with torch.no_grad():
        if isinstance(model, DecoderModel):
            tokens = model(x_test)
            if tokenization == "normalized":
                pred = torch.tensor([detokenize_normalized(t) for t in tokens], dtype=torch.float32).view(-1, 1).to(device)
            else:
                pred = torch.tensor([detokenize_unnormalized(t) for t in tokens], dtype=torch.float32).view(-1, 1).to(device)
        elif isinstance(model, MixtureDensityModel):
            pi, mu, sigma = model(x_test)
            pred = (pi * mu).sum(dim=-1, keepdim=True)  # Mean prediction
        else:
            pred = model(x_test)
    mse = mean_squared_error(y_test.cpu().numpy(), pred.cpu().numpy())
    return mse, pred
