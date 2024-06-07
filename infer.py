import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore

from tools.data import get_dataloaders
from tools.infer import make_prediction
from tools.model import ConvNet
from tools.train import eval_model
from config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"Available device: {device}")

    dl_train, dl_test = get_dataloaders(
        cfg.data.root,
        cfg.data.train_batch_size,
        cfg.data.test_batch_size,
        cfg.data.num_workers,
    )

    model_filename = cfg.model.save_to
    print(f"Loading model from '{model_filename}'")
    conv_network = ConvNet(use_batchnorm=cfg.model.use_batchnorm)
    conv_network.load_state_dict(torch.load(model_filename))
    conv_network.to(device)

    print("Evaluating model")
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss, train_accuracy = eval_model(conv_network, loss_fn, dl_train, device)
    test_loss, test_accuracy = eval_model(conv_network, loss_fn, dl_test, device)
    print(
        f"Loss (Train/Test): {train_loss:.3f}/{test_loss:.3f}."
        f" Accuracy, % (Train/Test): {train_accuracy:.2f}/{test_accuracy:.2f}"
    )

    print("Making test predictions")
    prediction = make_prediction(conv_network, dl_test, device)
    prediction_df = pd.DataFrame(
        {"image_id": np.arange(prediction.shape[0]), "label_id": prediction}
    )
    prediction_filename = cfg.inferring.prediction_filename
    prediction_df.to_csv(prediction_filename, index=False)
    print(f"Predictions saved in '{prediction_filename}'")


if __name__ == "__main__":
    main()