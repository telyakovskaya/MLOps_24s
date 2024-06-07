import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from mlflow import log_artifact, log_metric


def make_log_image(images, true_indices, pred_indices, n_epoch):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    tmean, tstd = (
        np.array([0.4914, 0.4822, 0.4465]),
        np.array([0.2023, 0.1994, 0.201]),
    )
    inverse_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(mean=-tmean / tstd, std=1.0 / tstd),
            torchvision.transforms.ToPILImage(),
        ]
    )
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(7, 35))
    for i in range(n):
        axes[i].imshow(inverse_transform(images[i]))
        true_label = classes[true_indices[i]]
        pred_label = classes[pred_indices[i]]
        axes[i].set_title(f"True : {true_label} Predicted {pred_label}")
    figname = f"visualization_after_{n_epoch}_epochs.png"
    fig.savefig(figname)
    log_artifact(figname)


def eval_model(network, loss_fn, dataloader, device, need_log_images=False, n_epoch=0):
    """
    returns: Среднее значение функции потерь и точности по батчам
    """
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for batch_idx, batch_data in enumerate(dataloader):
            X, y = batch_data
            X = X.to(device)
            y = y.to(device)
            preds = network(X)
            if need_log_images and batch_idx == 0:
                preds_indices = np.argmax(preds, axis=1)
                n = min(10, X.shape[0])
                make_log_image(X[:n], y[:n], preds_indices[:n], n_epoch)
            losses.append(loss_fn(preds, y).item())
            accuracies.append(
                torch.sum(preds.max(dim=1).indices == y).item() / y.shape[0]
            )
    return (np.mean(losses), np.mean(accuracies) * 100)


def training_loop(n_epochs, network, loss_fn, optimizer, dl_train, dl_test, device):
    """
    :param int n_epochs: Число итераций оптимизации
    :param torch.nn.Module network: Нейронная сеть
    :param Callable loss_fn: Функция потерь
    :param torch.nn.Optimizer optimizer: Оптимизатор
    :param torch.utils.data.DataLoader dl_train:
        Даталоадер для обучающей выборки
    :param torch.utils.data.DataLoader dl_test: Даталоадер для тестовой выборки
    :param torch.Device device: Устройство, на котором будут
        происходить вычисления
    :returns: Списки значений функции потерь и точности
        на обучающей и тестовой выборках
    """
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    for epoch in range(n_epochs):
        network.train()
        for images, labels in dl_train:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = network(images)
            loss = loss_fn(preds, labels)
            log_metric("loss", loss)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            train_loss, train_accuracy = eval_model(network, loss_fn, dl_train, device)
            log_metric("mean_train_loss", train_loss)
            log_metric("mean_train_accuracy", train_accuracy)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            test_loss, test_accuracy = eval_model(
                network, loss_fn, dl_test, device, True, epoch + 1
            )
            log_metric("mean_test_loss", test_loss)
            log_metric("mean_test_accuracy", test_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(
                f"Epoch {epoch + 1}/{n_epochs}: "
                f"Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}. "
                f"Accuracy, % (Train/Test): {train_accuracy:.2f}/"
                f"{test_accuracy:.2f}"
            )

    return train_losses, test_losses, train_accuracies, test_accuracies