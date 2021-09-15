# ITU - BLG 561E Deep Learning - Fall/2020
`Ozan Güldali & Cihat Kırankaya`

## Interim Project

## DETECTION OF AUTISM SPECTRUM DISORDER BASED ON DEEP FEATURES EXTRACTED FROM CONVOLUTIONAL NEURAL NETWORK MODEL

To run only ML, only CNN or both as transfer learning, app.py file can be run with corresponding function parameters.

- transfer_learning: True if wanted to transfer deep features from CNN model 
- method: "ML" or "CNN". Required if transfer_learning is False
- ml_model_name: "svm", "knn", "lr" or "all". Required if method is not CNN
- cv: any positive integer
- dataset_folder: "dataset"
- pretrain_file: pth file name without "pth" extension
- batch_size: any positive integer
- img_size: 112
- num_workers: 4
- cnn_model_name: "alexnet", "resnet18", "vgg16", "vgg19" or "densenet169". Required if method is not ML.
- optimizer_name: "Adam", "Padam" or "SGD"
- validation_freq: any positive rational number
- lr: any positive rational number
- momentum: any positive rational number
- partial: any positive rational number
- betas: any set of two positive rational numbers as (b1, b2)
- weight_decay: any positive rational number
- update_lr: True if wanted to periodically decrease the learning rate
- fine_tune: True if wanted to freeze first convolution block on CNN models
- num_epochs: any positive integer       
- normalize: True if wanted to normalize the data
- seed: 17

_Example of Transfer Leaning:_
- Unless exists, 84.35_PreTrained_resnet18_Adam_dataset_out.pth file must be downloaded and inserted into "cnn" directory.
- Link to file: https://github.com/ozanguldali/interim-brain-mri/blob/master/cnn/84.35_PreTrained_resnet18_Adam_dataset_out.pth
`app.main(transfer_learning=True, ml_model_name="lr", cnn_model_name="resnet18", is_pre_trained=True,
         dataset_folder="dataset", pretrain_file="84.35_PreTrained_resnet18_Adam_dataset_out", seed=17)`