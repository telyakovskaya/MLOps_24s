import torch
import numpy as np
import pandas as pd

from model import ConvNet
from train import eval_model, training_loop
from data import dataset_cifar10_train, dataset_cifar10_test


def make_prediction(network, dataloader):
    predictions = []
    network.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            probs = network(X).to(torch.device('cpu')).numpy()
            preds = np.argmax(probs, axis=1)
            predictions.append(preds)
    return np.concatenate(predictions)


if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    print(f'Available device: {device}')
    
    batch_size = 1024
    test_size = len(dataset_cifar10_test)
    dl_test = torch.utils.data.DataLoader(dataset_cifar10_test, batch_size=batch_size, num_workers=2)
    
    model_filename = 'classifier.pth'
    print(f'Loading model from \'{model_filename}\'')
    conv_network = ConvNet(use_batchnorm=True)
    conv_network.load_state_dict(
        torch.load(model_filename)
    )
    conv_network.to(device)

    print('Making test predictions')
    prediction = make_prediction(conv_network, dl_test)

    prediction_df = pd.DataFrame({'image_id': np.arange(test_size), "label_id": prediction})
    prediction_filename = 'test_prediction.csv'
    prediction_df.to_csv(prediction_filename, index=False)
    print(f'Predictions saved in \'{prediction_filename}\'')