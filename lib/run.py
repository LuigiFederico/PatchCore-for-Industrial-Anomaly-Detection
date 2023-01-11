from .data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from .patch_core import PatchCore
from .utils import backnones

ALL_CLASSES = mvtec_classes()


def run_model(
    classes: list = ALL_CLASSES, 
    f_coreset: float = 0.1,
    vanilla: bool = True,
    backbone: str = 'WideResNet50'):

    results = {}  # key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]
    if vanilla:
        size = DEFAULT_SIZE
    elif backbone == 'ResNet50':  # NON RICORDO IL NOME GIUSTO
        size = 448
    else:  # ViTB...
        size = 224

    print(f'Running PatchCore...')
    for cls in classes:
        train_dl, test_dl = MVTecDataset(cls, size=size).get_dataloaders()
        patch_core = PatchCore(f_coreset, vanilla=vanilla, backbone=backnones[backbone])

        print(f'\nClass {cls}:')
        print(f'Training...')
        patch_core.fit(train_dl)

        print(f'Testing...')
        image_rocauc, pixel_rocauc = patch_core.evaluate(test_dl)

        print(f'Results:')
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        print(f'- image-level ROC AUC = {image_rocauc:.2f}')
        print(f'- pixel-level ROC AUC = {pixel_rocauc:.2f}\n')
    
    # Save global results and statistics
    image_results = [v[0] for k, v in results.items()]
    average_image_rocauc = sum(image_results) / len(image_results)
    pixel_resuts = [v[1] for k, v in results.items()]
    average_pixel_rocauc = sum(pixel_resuts) / len(pixel_resuts)

    print(f'- Average image-level ROC AUC = {average_image_rocauc:.2f}\n')
    print(f'- Average pixel-level ROC AUC = {average_pixel_rocauc:.2f}\n')



    # Display global results






if __name__ == "__main__":
    run_model(vanilla=True, backbone='WideResNet50')
