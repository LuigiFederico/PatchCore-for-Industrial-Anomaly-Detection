import os
from os.path import isdir
import tarfile
import wget
import ssl
from pathlib import Path
from PIL import Image
from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Parameters
DATASETS_PATH = Path("./datasets")
IMAGENET_MEAN = tensor([.485, .456, .406])  # Standard Parameter that can be found online
IMAGENET_STD = tensor([.229, .224, .225])  # Standard Parameter that can be found online
DEFAULT_SIZE = 224
DEFAULT_RESIZE = 256
class_links = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz"
}


def mvtec_classes():
    return [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile",
        "toothbrush", "transistor", "wood", "zipper"]


class MVTecDataset:
    def __init__(self, cls: str, size: int = DEFAULT_SIZE):
        self.cls = cls
        self.size = size
        if cls in mvtec_classes():
            self.check_and_download_cls()
        self.train_ds = MVTecTrainDataset(cls, size)
        self.test_ds = MVTecTestDataset(cls, size)

    def check_and_download_cls(self):
        if not isdir(DATASETS_PATH / self.cls):
            print(f"Class '{self.cls}' has not been found in '{DATASETS_PATH}/'. Downloading... \n")
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(class_links[self.cls])
            with tarfile.open(f"{self.cls}.tar.xz") as tar:
                tar.extractall(DATASETS_PATH)
            os.remove(f"{self.cls}.tar.xz")
            print(f"Correctly Downloaded \n")
        else:
            print(f"Class '{self.cls}' has been found in '{DATASETS_PATH}/'\n")

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


class MVTecTrainDataset(ImageFolder):
    def __init__(self, cls: str, size: int, resize: int = DEFAULT_RESIZE):
        super().__init__(
            root=DATASETS_PATH / cls / "train",
            transform=transforms.Compose([    # Transform img composing several actions
                transforms.Resize(resize),    # Resize the image to the default value of 256 if not changed
                transforms.CenterCrop(size),  # Center the image
                transforms.ToTensor(),        # Transform the image into a tensor
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # Normalize the image
            ])
        )
        self.cls = cls
        self.size = size


class MVTecTestDataset(ImageFolder):
    def __init__(self, cls: str, size: int, resize: int = DEFAULT_RESIZE):
        super().__init__(
            root=DATASETS_PATH / cls / "test",
            transform=transforms.Compose([    # Transform img composing several actions
                transforms.Resize(resize),    # Resize the image to the default value of 256 if not changed
                transforms.CenterCrop(size),  # Center the image
                transforms.ToTensor(),        # Transform the image into a tensor
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # Normalize the image
            ]),
            target_transform=transforms.Compose([  # Transform mask composing several actions
                transforms.Resize(resize),         # Resize the mask to the default value of 256 if not changed
                transforms.CenterCrop(size),       # Center the mask
                transforms.ToTensor(),             # Transform the mask into a tensor
            ]),
        )
        self.cls = cls
        self.size = size

    def getitem(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:  # In this way is possible to understand the class label of the image
            target = Image.new('L', (self.size, self.size))  # L is equal to 8-bit pixels black and white
            sample_class = 0
        else:
            target_path = path.replace("test", "ground_truth")  # Change folder and goes into mask folder
            target_path = target_path.replace(".png", "_mask.png")  # Change extension required
            target = self.loader(target_path)
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)  # Apply transformation to the image

        if self.target_transform is not None:
            target = self.target_transform(target)  # Apply transformation to the mask

        return sample, target[:1], sample_class
