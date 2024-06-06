import torch
import numpy as np

from model import ConvNet
from data import dataset_cifar10_train, dataset_cifar10_test


def eval_model(network, loss_fn, dataloader, device):
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            preds = network(X)
            losses.append(loss_fn(preds, y).item())
            accuracies.append(torch.sum(preds.max(dim=1).indices == y).item() / y.shape[0])
    return (np.mean(losses), np.mean(accuracies) * 100)


def training_loop(n_epochs, network, loss_fn, optimizer, dl_train, dl_test, device):
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    for epoch in range(n_epochs):
        network.train()
        for images, labels in dl_train:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = network(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            train_loss, train_accuracy = eval_model(network, loss_fn, dl_train, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_loss, test_accuracy = eval_model(network, loss_fn, dl_test, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f'Epoch {epoch + 1}/{n_epochs}: Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}. Accuracy, % (Train/Test): {train_accuracy:.2f}/{test_accuracy:.2f}')

    return train_losses, test_losses, train_accuracies, test_accuracies


if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    print(f'Available device: {device}')
    
    batch_size = 512
    dl_train = torch.utils.data.DataLoader(dataset_cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_test = torch.utils.data.DataLoader(dataset_cifar10_test, batch_size=batch_size, num_workers=2)
    conv_network = ConvNet(use_batchnorm=True)
    conv_network.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(conv_network.parameters(), lr=2e-2)

    n_epochs = 2
    print(f'Training (n_epochs: {n_epochs})')
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=n_epochs, network=conv_network, loss_fn=loss_fn, 
        optimizer=optimizer, dl_train=dl_train, dl_test=dl_test, device=device
    )

    print(f'Evaluating model')
    train_loss, train_accuracy = eval_model(conv_network, loss_fn, dl_train, device)
    test_loss, test_accuracy = eval_model(conv_network, loss_fn, dl_test, device)
    print(f'Finally: Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}. Accuracy, % (Train/Test): {train_accuracy:.2f}/{test_accuracy:.2f}')

    filename = 'classifier.pth'
    torch.save(conv_network.state_dict(), filename)
    print(f'Model saved in \'{filename}\'')