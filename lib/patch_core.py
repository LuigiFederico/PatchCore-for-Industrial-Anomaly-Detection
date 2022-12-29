import torch
from torch.nn import functional as F
import numpy as np

class KNNextractor():
    pass


class PatchCore(torch.nn.Module):
    def __init__(
            self,
            MVTecDataset,
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

        # Memory bank
        self.memory_bank = []

        # ALTRO?????????????????????????????????????????????????????????????????????

        raise NotImplementedError


    def forward(self, sample):
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


    def fit(self, train_dataloader):
        
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
        
        self.memory_bank = torch.cat(self.memory_bank, 0) # Stack the patches

        # Coreset subsampling
        
        # TO BE CONTINUED...





        raise NotImplementedError