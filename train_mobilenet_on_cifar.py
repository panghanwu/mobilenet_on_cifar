from torch import nn
from torch.optim import SGD

from cifars import get_cifar100_datasets
from mobilenet import create_mobilenet_for_cifar, MOBILENET_LARGE_CONFIG, MOBILENET_LARGE_VTAPER_CONFIG
from trainer import Trainer
from utils import init_logging_configs

DATASET_DIR: str = 'datasets'
DEVICE: str = 'cpu'
BATCH_SIZE: int = 64
EPOCHS: int = 200
DROPOUT: float = 0.1
DEBUGGING = True

init_logging_configs(DEBUGGING)
model = create_mobilenet_for_cifar(num_classes=10, configs=MOBILENET_LARGE_CONFIG)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
train_loader, test_loader = get_cifar100_datasets(BATCH_SIZE, DATASET_DIR)

trainer = Trainer(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion=nn.CrossEntropyLoss(),
    device=DEVICE,
    mission_name='cifar_mobilenet',
    stopping_patience=0,
    debugging=DEBUGGING
)

trainer.fit(EPOCHS)