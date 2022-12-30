from data import MVTecDataset, mvtec_classes
from patch_core import PatchCore

ALL_CLASSES = mvtec_classes()


def run_model(
    classes: list = ALL_CLASSES, 
    f_coreset: float = 0.1,
):
    print(f'Running PatchCore...')
    for cls in classes:
        train_dl, test_dl = MVTecDataset(cls).get_dataloaders()
        patch_core = PatchCore(f_coreset)

        print(f'Class {cls}:')
        print(f'  Training...')
        patch_core.fit(train_dl)
        print(f'Testing...')
        # evaluate..

        # Print test results
        # Save test results
    
    # Save global results

    # Return global results


if __name__ == "__main__":
    run_model()
