import torch
import random
import clip  #needed for CLIP
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from torch import tensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from .utils import gaussian_blur, get_coreset
from PIL import Image  #needed for CLIP

class PatchCore(torch.nn.Module):
    def __init__(
            self,
            f_coreset:float = 0.01,   # Fraction rate of training samples
            eps_coreset: float = 0.90, # SparseProjector parameter
            k_nearest: int = 3,        # k parameter for K-NN search
            vanilla: bool = True,
            backbone: str = 'wide_resnet50_2',
            image_size: int =224
    ):
        assert f_coreset > 0
        assert eps_coreset > 0
        assert k_nearest > 0


        super(PatchCore, self).__init__()

        # Define hooks to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        # Setup backbone net
        print(f"Vanilla Mode {vanilla}")
        print(f"Net Used {backbone}")

        if vanilla==True:
            self.model = torch.hub.load('pytorch/vision', 'wide_resnet50_2', pretrained=True)
            self.model.layer2[-1].register_forward_hook(hook)  # Register hooks
            self.model.layer3[-1].register_forward_hook(hook)  # Register hooks
        else:
            self.model, _ = clip.load(backbone, device="cpu")
            if "ViT" in backbone:
                NotImplementedError()
                # Non ci sono layer
            else:
                self.model.visual.layer2[-1].register_forward_hook(hook)  # Register hooks
                self.model.visual.layer3[-1].register_forward_hook(hook)  # Register hooks

        self.model.eval()
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False


        # Parameters
        self.memory_bank = []
        self.f_coreset = f_coreset
        self.eps_coreset = eps_coreset
        self.k_nearest = k_nearest
        self.vanilla = vanilla
        self.backbone = backbone
        self.image_size = image_size


    def forward(self, sample: tensor):

        """
            Initialize self.features and let the input sample passing
            throught the backbone net self.model.
            The registered hooks will extract the layer 2 and 3 feature maps.
            Return:
                self.feature filled with extracted feature maps
        """


        self.features = []
        if self.vanilla:
            _ = self.model(sample)
        else:
            _ = self.model.visual(sample)  #Clip

        return self.features


    def fit(self, train_dataloader :DataLoader, scale :int=1) -> None:

        """
            Training phase
            Creates memory bank from train dataset and apply greedy coreset subsampling.
        """
        tot=int(len(train_dataloader)/scale)
        counter=0

        for sample, _ in tqdm(train_dataloader, total=tot):
            feature_maps = self(sample)  # Extract feature maps

            # Create aggregation function of feature vectors in the neighbourhood
            self.avg = torch.nn.AvgPool2d(3, stride=1)
            fmap_size = feature_maps[0].shape[-2]  # Feature map sizes h, w
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

            # Create patch
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)          # Merge the resized feature maps
            patch = patch.reshape(patch.shape[1], -1).T # Craete a column tensor

            self.memory_bank.append(patch) # Fill memory bank
            counter+=1
            if counter > tot:
                break

        self.memory_bank = torch.cat(self.memory_bank, 0) # VStack the patches

        # Coreset subsampling
        if self.f_coreset < 1:
            coreset_idx = get_coreset(
                self.memory_bank,
                l = int(self.f_coreset * self.memory_bank.shape[0]),
                eps = self.eps_coreset
            )
            self.memory_bank = self.memory_bank[coreset_idx]

    def evaluate(self, test_dataloader: DataLoader):
        """
            Compute anomaly detection score and relative segmentation map for
            each test sample. Returns the ROC AUC computed from predictions scores.

            Returns:
            - image-level ROC-AUC score
            - pixel-level ROC-AUC score
        """


        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        transform = T.ToPILImage()

        for sample, mask, label in tqdm(test_dataloader):

            image_labels.append(label)
            pixel_labels.extend(mask.flatten().numpy())

            score, segm_map = self.predict(sample)  # Anomaly Detection

            image_preds.append(score.numpy())
            pixel_preds.extend(segm_map.flatten().numpy())

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)

        # Compute ROC AUC for prediction scores
        image_level_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_level_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        return image_level_rocauc, pixel_level_rocauc


    def predict(self, sample: tensor):
        """
            Anomaly detection over a test sample
            1. Create a locally aware patch feature of the test sample.
            2. Compute the image-level anomaly detection score by comparing
            the test patch with the nearest neighbours patches inside the memory bank.
            3. Compute a segmentation map realigning computed path anomaly scores based on
            their respective spacial location. Then upscale the segmentation map by
            bi-linear interpolation and smooth the result with a gaussian blur.
            Args:
            - sample:  test sample
            Returns:
            - Segmentation score
            - Segmentation map
        """

        # Patch extraction
        feature_maps = self(sample)
        resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        # Compute maximum distance score s* (equation 6 from the paper)
        distances = torch.cdist(patch, self.memory_bank,
                                p=2.0)  # L2 norm dist btw test patch with each patch of memory bank
        dist_score, dist_score_idxs = torch.min(distances,
                                                dim=1)  # Val and index of the distance scores (minimum values of each row in distances)
        s_idx = torch.argmax(dist_score)  # Index of the anomaly candidate patch
        s_star = torch.max(dist_score)  # Maximum distance score s*
        m_test_star = torch.unsqueeze(patch[s_idx], dim=0)  # Anomaly candidate patch
        m_star = self.memory_bank[dist_score_idxs[s_idx]].unsqueeze(
            0)  # Memory bank patch closest neighbour to m_test_star

        # KNN
        knn_dists = torch.cdist(m_star, self.memory_bank,
                                p=2.0)  # L2 norm dist btw m_star with each patch of memory bank
        _, nn_idxs = knn_dists.topk(k=self.k_nearest,
                                    largest=False)  # Values and indexes of the k smallest elements of knn_dists

        # Compute image-level anomaly score s
        m_star_neighbourhood = self.memory_bank[nn_idxs[0, 1:]]
        w_denominator = torch.linalg.norm(m_test_star - m_star_neighbourhood,
                                          dim=1)  # Sum over the exp of l2 norm distances btw m_test_star and the m* neighbourhood
        norm = torch.sqrt(
            torch.tensor(patch.shape[1]))  # Softmax normalization trick to prevent exp(norm) from becoming infinite
        w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))  # Equation 7 from the paper
        s = w * s_star

        # Segmentation map
        fmap_size = feature_maps[0].shape[-2:]  # Feature map sizes: h, w
        segm_map = dist_score.view(1, 1, *fmap_size)  # Reshape distance scores tensor
        segm_map = torch.nn.functional.interpolate(
            # Upscale by bi-linaer interpolation to match the original input resolution
            segm_map,
            size=(self.image_size, self.image_size),
            mode='bilinear'
        )
        segm_map = gaussian_blur(segm_map)  # Gaussian blur of kernel width = 4

        return s, segm_map
