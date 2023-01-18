from .data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from .patch_core import PatchCore
from .utils import backnones, dataset_scale_factor

ALL_CLASSES = mvtec_classes()


def run_model(
    classes: list = ALL_CLASSES,
    f_coreset: float = 0.1,
    vanilla: bool = True,
    backbone: str = 'WideResNet50'):

    if (backbone == "WideResNet50" and vanilla != True) or (backbone != "WideResNet50" and vanilla == True):
        raise Exception('Please check the Vanilla and Backbone values. You can use Vanilla == True just with WideResNet50 architecture.')

    results = {}  # key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]
    if vanilla:
        size = DEFAULT_SIZE
    elif backbone == 'ResNet50':  # RN50
        size = 224
    elif backbone == 'ResNet50-4':  # RN50x4
        size = 288
    elif backbone == 'ResNet50-16':  # RN50x16
        size = 384
    elif backbone == 'ResNet101':  # RN50x101
        size = 224
    else:
        raise Exception('You can use the following nets: ResNet50, ResNet50-4, ResNet50-16, ResNet101')

    print(f'Running PatchCore...')
    for cls in classes:
        train_dl, test_dl = MVTecDataset(cls, size=size, vanilla=vanilla, backbone=backnones[backbone]).get_dataloaders()
        patch_core = PatchCore(f_coreset, vanilla=vanilla, backbone=backnones[backbone], image_size=size)

        print(f'\nClass {cls}:')
        print(f'Training...')
        patch_core.fit(train_dl, scale=dataset_scale_factor[backbone])

        print(f'Testing...')
        image_rocauc, pixel_rocauc = patch_core.evaluate(test_dl)

        print(f'Results:')
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        print(f'- image-level ROC AUC = {image_rocauc:.3f}')
        print(f'- pixel-level ROC AUC = {pixel_rocauc:.3f}\n')

    # Save global results and statistics
    image_results = [v[0] for k, v in results.items()]
    average_image_rocauc = sum(image_results) / len(image_results)
    pixel_resuts = [v[1] for k, v in results.items()]
    average_pixel_rocauc = sum(pixel_resuts) / len(pixel_resuts)

    print(f'- Average image-level ROC AUC = {average_image_rocauc:.3f}\n')
    print(f'- Average pixel-level ROC AUC = {average_pixel_rocauc:.3f}\n')



    # Display global results






if __name__ == "__main__":
    run_model(vanilla=True, backbone='WideResNet50')
