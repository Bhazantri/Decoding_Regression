import requests
import pandas as pd
import os

# Download UCI datasets (example)
datasets = {
    "airfoil": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
}

os.makedirs("data", exist_ok=True)
for name, url in datasets.items():
    response = requests.get(url)
    with open(f"data/{name}.csv", "wb") as f:
        f.write(response.content)
    print(f"Downloaded {name}")
