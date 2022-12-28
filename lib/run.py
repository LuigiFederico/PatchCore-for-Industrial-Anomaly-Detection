from data import MVTecDataset, mvtec_classes
from typing import List

ALL_CLASSES = mvtec_classes()


def run_model(classes: List = ALL_CLASSES):
    for cls in classes:
        train_ds, test_ds = MVTecDataset(cls).get_dataloaders()


if __name__ == "__main__":
    run_model()
