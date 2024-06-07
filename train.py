import subprocess

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore

from tools.data import get_dataloaders
from tools.model import ConvNet
from tools.train import training_loop
from config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"Available device: {device}")

    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_name)
    mlflow.set_tracking_uri(cfg.mlflow.uri)

    git_commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )
    mlflow.log_param("git_commit_id", git_commit_id)
    mlflow.log_param("learning_rate", cfg.training.learning_rate)
    mlflow.log_param("n_epochs", cfg.training.epochs)

    dl_train, dl_test = get_dataloaders(
        cfg.data.root,
        cfg.data.train_batch_size,
        cfg.data.test_batch_size,
        cfg.data.num_workers,
    )
    conv_network = ConvNet(use_batchnorm=cfg.model.use_batchnorm)
    conv_network.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        conv_network.parameters(), lr=cfg.training.learning_rate
    )

    n_epochs = cfg.training.epochs
    print(f"Training (n_epochs: {n_epochs})")
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=n_epochs,
        network=conv_network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dl_train=dl_train,
        dl_test=dl_test,
        device=device,
    )

    filename = cfg.model.save_to
    torch.save(conv_network.state_dict(), filename)
    print(f"Model saved in '{filename}'")


if __name__ == "__main__":
    main()