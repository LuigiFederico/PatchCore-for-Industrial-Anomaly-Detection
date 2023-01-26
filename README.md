
# PatchCore for Industrial Anomaly Detection


[[Paper]]() [[Colab]](https://colab.research.google.com/drive/1mNQly_8bWsa208bxplfphZHMygwgt2Dh?usp=sharing) [[Presentation]](https://docs.google.com/presentation/d/1o0RiK4-u8XyQ-H9zaHd7XMw7L6_cquO0iL2jupXvTBk/edit?usp=sharing)

Nowadays the ability to recognize defective parts is very important in large-scale industrial manufacturing. The problem becomes more challenging when we try to fit a model using a nominal (non-defective) image only at training time. This problem is called "cold-start". 
The best approaches combine embeddings from ImageNet models with an outlier detection model. In this repository, we reproduce the state-of-the-art approach called [__PatchCore__](https://arxiv.org/abs/2106.08265) which uses a representative memory bank of nominal patch features to achieve high performances both on detection and localization. We developed this approach using the MVTec Anomaly Detection dataset on which PatchCore achieves an image-level anomaly detection AUROC score of up to 98.4\%, almost doubling the performance compared to related works. Furthermore, we tested the patchcore by replacing the backbone with an alternative solution, trying to get better generalizable representations to leverage for the downstream task. This alternative solution is represented by [__CLIP__](https://arxiv.org/abs/2103.00020): we exploited the pretrained Image Encoder of CLIP instead of the ImageNet pretrained one. The experiment results suggest that using ResNet50x16 as the architecture of the image encoder we obtain better results on a smaller training set.

## Authors  

- Carluccio Alex, s302373 [[GitHub]](https://github.com/LordAssalt)  
- Federico Luigi, s295740 [[GitHub]](https://github.com/LuigiFederico)  
- Longo Samuele, s305202 [[GitHub]](https://github.com/Clyde99x)  


## Approach 

![PatchCore approach](PatchCore_Architecture.png)

[Source](https://github.com/amazon-science/patchcore-inspection/blob/main/images/architecture.png)

## Usage from Colab

You can easly run our code on the provided [Colab Jupyther Notebook](https://colab.research.google.com/drive/1mNQly_8bWsa208bxplfphZHMygwgt2Dh?usp=sharing). We suggest to use a GPU runtime.

First of all, install all the required frameworks and import our repository. Then, you need to __choose which backbone you want to use__ as _Pretrained Encoder_ to run the PatchCore method.
If you want to run the vanilla version of PatchCore, use `WideResNet50` as backbone. If you want to test the CLIP version, then you can choose between the following architectures:

- `ResNet50`
- `ResNet50-4`
- `ResNet50-16`
- `ResNet101`

You can easily verify the available backbones by running the function:

```Python
    display_backbones()
```  


It is possible to __select which MVTec classes to run the PatchCore__. You can list the available classes using the function:

```Python
    display_MVTec_classes()
``` 

If you want to run the PatchCore algorithm over the entire MVTec dataset, set the variable `classes` to `None`. 

Finally, you can __run the model__ through the API function 

```Python
    run_model(classes: list, backbone: str)
``` 

It will automatically download the dataset classes and it will perform three phases:
- __Training phase__: locally-aware patch feature creation.
- __Coreset subsampling__: reduces the number of patches inside the memory back.
- __Testing phase__: computes image-level anomaly detection and segmentation.

As a result, the __AUROC__ score calculated using the obtained AD scores and the MVTec test dataset will be displayed. If you run multiple classes then you will observe the AUROC for each class and, at the end, the average AUROC score.
