"SR-AttNet: An interpretable Stretch–Relax attention based deep neuralnetwork for polyp segmentation in colonoscopy images"

Test Instances: gives the image IDs for the test cases (the rest are the train cases)

Weights:
    1. 256x256_trainedOn(kvseg)_loss(dice+bce).h5   : Trained on Kvasir-Seg dataset, img dimension 256, loss= bce+dice
    2. 256x256_trainedOn(ClinicDB)_loss(dice).h5    : Trained on CVC-ClinicDB dataset, img dimension 256, loss= dice
