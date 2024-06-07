from dataclasses import dataclass


@dataclass
class Data:
    train_batch_size: int
    test_batch_size: int
    root: str
    num_workers: int


@dataclass
class Model:
    save_to: str
    use_batchnorm: bool


@dataclass
class Training:
    epochs: int
    learning_rate: float


@dataclass
class Inferring:
    prediction_filename: str


@dataclass
class Mlflow:
    experiment_name: str
    uri: str


@dataclass
class Params:
    data: Data
    model: Model
    training: Training
    inferring: Inferring
    mlflow: Mlflow