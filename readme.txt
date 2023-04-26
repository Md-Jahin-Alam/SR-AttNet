"SR-AttNet: An interpretable Stretch-Relax Attention based Deep Neural Network for Polyp Segmentation in Colonoscopy Images"
(link) https://www.sciencedirect.com/science/article/pii/S0010482523004109?via%3Dihub

Test Instances: gives the image IDs for the test cases (the rest are the train cases)

Weights: 
    (link) https://drive.google.com/drive/folders/1uM9cstAZdLKFPZNv4y-bd7m9Tf4OsjVA?usp=share_link
    1. 256x256_trainedOn(kvseg)_loss(dice+bce).h5   : Trained on Kvasir-Seg dataset, img dimension 256, loss= bce+dice
    2. 256x256_trainedOn(ClinicDB)_loss(dice).h5    : Trained on CVC-ClinicDB dataset, img dimension 256, loss= dice
