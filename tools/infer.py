import numpy as np
import torch


def make_prediction(network, dataloader, device):
    predictions = []
    network.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            probs = network(X).to(torch.device("cpu")).numpy()
            preds = np.argmax(probs, axis=1)
            predictions.append(preds)
    return np.concatenate(predictions)