from .data import MVTecDataset, mvtec_classes
from .patch_core import PatchCore

ALL_CLASSES = mvtec_classes()


def run_model(
    classes: list = ALL_CLASSES, 
    f_coreset: float = 0.1,
):
    results = {}  # key = class, Value = [image-level ROC AUC, pixel-level ROC AUC]

    print(f'Running PatchCore...')
    for cls in classes:
        train_dl, test_dl = MVTecDataset(cls).get_dataloaders()
        patch_core = PatchCore(f_coreset)  # <--- UN PATCHCORE PER OGNI CLASSE???????????????????????????????

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

    # Display global results






if __name__ == "__main__":
    run_model()
