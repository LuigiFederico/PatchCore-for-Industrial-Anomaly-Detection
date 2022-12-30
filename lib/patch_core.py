import torch
from torch import tensor
from torch.nn import functional as F
import numpy as np

from utils import get_coreset

class KNNextractor():
    pass


class PatchCore(torch.nn.Module):
    def __init__(
            self,
            f_coreset: float = 0.01,   # Fraction rate of training samples
            eps_coreset: float = 0.09, # SparseProjector parameter
            ):
        super(PatchCore, self).__init__()
        
        # Define hooks to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        # Setup backbone net
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)   
        
        # Disable gradient computation
        for param in self.model.parameters(): 
            param.requires_grad = False
        self.model.eval()

        # Register hooks
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        # Parameters
        self.memory_bank = []
        self.f_coreset = f_coreset
        self.eps_coreset = eps_coreset

        # ALTRO?????????????????????????????????????????????????????????????????????

        raise NotImplementedError


    def forward(self, sample: tensor) -> list(tensor):
        """
            Initialize self.features and let the input sample passing
            throught the backbone net self.model. 
            The registered hooks will extract the layer 2 and 3 feature maps.

            Return:
                self.feature filled with extracted feature maps
        """
        self.features = []
        _ = self.model(sample)
        return self.features


    def fit(self, train_dataloader) -> None:
        """ 
            Training phase 

            Creates memory bank from train dataset and apply greedy coreset subsampling.
        """
        for sample, _ in train_dataloader:
            feature_maps = self(sample)  # Extract feature maps

            # Create aggregation function of feature vectors in the neighbourhood
            self.avg = torch.nn.AvgPool2d(3, stride=1)
            fmap_size = feature_maps[0].shape[-2:]  # Feature map sizes: h, w
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size) 
            
            # Create patch
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)          # Merge the resized feature maps
            patch = patch.reshape(patch.shape[1], -1).T # Craete a column tensor
            
            self.memory_bank.append(patch) # Fill memory bank
        
        self.memory_bank = torch.cat(self.memory_bank, 0) # VStack the patches

        # Coreset subsampling
        if self.f_coreset < 1:
            coreset_idx = get_coreset(
                self.memory_bank,
                l = int(self.f_coreset * self.memory_bank.shape[0]),
                eps = self.eps_coreset
            )
            self.memory_bank = self.memory_bank[coreset_idx]
       


    def evaluate(self, test_dataloader):
        raise NotImplementedError


    def predict(self):
        raise NotImplementedError