import torch


def make_features(cfg, use_batchnorm):
    ret = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            ret += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            ret += [torch.nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)]
            if use_batchnorm:
                ret += [torch.nn.BatchNorm2d(v)]
            ret += [torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*ret)


class ConvNet(torch.nn.Module):
    cfg = [32, "M", 64, 64, "M", 128, 128, "M"]
    
    def __init__(self, n_classes=10, use_batchnorm=False, dropout_p=0.0):
        super().__init__()
        self.n_classes = n_classes
        self.features = make_features(self.cfg, use_batchnorm)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.Dropout1d(dropout_p, inplace=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.Dropout1d(dropout_p, inplace=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.classifier(self.avgpool(self.features(x)).reshape((-1, 512)))
        return y