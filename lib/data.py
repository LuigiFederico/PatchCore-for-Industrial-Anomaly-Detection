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
DEFAULT_SIZE = 224
DEFAULT_RESIZE = 256

IMAGENET_MEAN = tensor([.485, .456, .406])  
IMAGENET_STD = tensor([.229, .224, .225]) 
CLIP_MEAN = tensor([.481, .457, .408])
CLIP_STD = tensor([.268, .261, .275])

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
        "bottle", 
        "cable", 
        "capsule", 
        "carpet", 
        "grid", 
        "hazelnut", 
        "leather", 
        "metal_nut", 
        "pill", 
        "screw", 
        "tile",
        "toothbrush", 
        "transistor", 
        "wood", 
        "zipper"
    ]


class MVTecDataset:
    def __init__(
            self, 
            cls: str, 
            size: int = DEFAULT_SIZE, 
            vanilla: bool = True,
    ):
        assert cls in mvtec_classes()

        # Parameters
        self.cls = cls
        self.size = size  
        
        # Download data
        self.check_and_download_cls()

        # Build datasets
        if vanilla:
            resize = DEFAULT_RESIZE
        else:   # CLIP
            resize = size

        self.train_ds = MVTecTrainDataset(cls, size, resize, vanilla)
        self.test_ds = MVTecTestDataset(cls, size, resize, vanilla)


    def check_and_download_cls(self):
        """
            If the expected dataset path is not found, 
            download the dataset inside /dataset.
        """

        if not isdir(DATASETS_PATH / self.cls):
            print(f"Class '{self.cls}' has not been found in '{DATASETS_PATH}/'. Downloading... \n")
            
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(class_links[self.cls]) # Downoad of the zipped dataset
            with tarfile.open(f"{self.cls}.tar.xz") as tar: # Unzip
                tar.extractall(DATASETS_PATH)
            os.remove(f"{self.cls}.tar.xz") # Clean up
            
            print(f"Correctly Downloaded \n")

        else:
            print(f"Class '{self.cls}' has been found in '{DATASETS_PATH}/'\n")


    def get_datasets(self):
        """
            Returns as tuple:
            - train dataset (MVTecTrainDataset class)
            - test dataset (MVTecTestDataset class)
        """
        return self.train_ds, self.test_ds


    def get_dataloaders(self):
        """
            Returns as tuple:
            - train dataloader (torch.utils.data.DataLoader class)
            - test dataloader (torch.utils.data.DataLoader class)
        """
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class MVTecTrainDataset(ImageFolder):
    def __init__(
            self, 
            cls: str, 
            size: int, 
            resize: int = DEFAULT_RESIZE, 
            vanilla: bool = True,
    ):
        # Vanilla/CLIP image pre-processing
        if vanilla:
            transform = transforms.Compose([        
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])     
        else:
            transform = transforms.Compose([
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])

        # Parameters
        super().__init__(
                root = DATASETS_PATH / cls / "train",
                transform = transform )
        self.cls = cls
        self.size = size


class MVTecTestDataset(ImageFolder):
    def __init__(
            self, 
            cls: str, 
            size: int, 
            resize: int = DEFAULT_RESIZE, 
            vanilla: bool = True,
    ):

        # Vanilla/CLIP image and mask pre-processing
        if vanilla:
            transform = transforms.Compose([         # Image transform
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),  
                transforms.ToTensor(),  
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
            ])
            target_transform = transforms.Compose([  # Mask transform
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(size), 
                transforms.ToTensor(),
            ])
        else:
            transform  = transforms.Compose([        # Image transform
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ])

            target_transform = transforms.Compose([  # Mask transform
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(size),
                _convert_image_to_rgb, 
                transforms.ToTensor(),
            ])

        # Parameters
        super().__init__(
            root=DATASETS_PATH / cls / "test",
            transform = transform,
            target_transform = target_transform
        )

        self.cls = cls
        self.size = size


    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:                                      # Nominal image
            mask = Image.new('L', (self.size, self.size))       # L -> 8-bit pixels black and white
            sample_class = 0
        else:                                                   # Anomaly image
            mask_path = path.replace("test", "ground_truth")    # Change folder and goes into mask folder
            mask_path = mask_path.replace(".png", "_mask.png")  # Change extension required
            mask = self.loader(mask_path)                       # Load the mask
            sample_class = 1

        # Trasnformations 
        if self.transform is not None:
            sample = self.transform(sample)  

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sample, mask[:1], sample_class
