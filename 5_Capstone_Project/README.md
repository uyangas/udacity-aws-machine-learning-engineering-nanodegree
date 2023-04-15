# Alzheimer's Severity Prediction from Brain X-ray Images (Capstone project)


## 1. Project Set Up and Installation

The following packages were used:
- `sagemaker` (lastest version)
- `boto3` (lastest version)
- `numpy` (lastest version)
- `torch` (lastest version)
- `torchvision` (lastest version)
- `argparse` (lastest version)
- `os` (lastest version)
- `smdebug` (lastest version)
- `logging` (lastest version)
- `sys` (lastest version)

Configuration:
- `instance_type`: `"ml.m5.2xlarge"` (model training)
- `instance_type`: `"ml.g4dn.xlarge"` (hyperparameter tuning)
- `instance_count`: `2`
- python version: `3.8`


## 2. Dataset
The dataset is Alzheimer's disease X-ray (2D image) dataset from Kaggle platform and can be accessed [here](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).

The dataset consists of `train` - `5,121` and `test` - `1,279` images, total of `6,400` images. The images were labeled as one of the following severity levels:
1. Mild Demented
2. Moderate Demented
3. Non Demented
4. Very Mild Demented

Out of `5,121` training image, `20%` were randomly sampled to create `valid` set to validate the model. 

## 3. Pre-trained Model Hyperparameter Tuning & Training

For pre-trained model, `VGG16` was chosen. Although [`VGG`](https://www.researchgate.net/figure/Comparison-of-Convolutional-Neural-Network-Architectures-in-Terms-of-Size-and-Performance_fig3_363738835) is relatively older and larger compared to the recent smaller models such as `Inception` and `ResNet`, it tends to be more [robust](https://www.eetimes.eu/ai-tradeoff-accuracy-or-robustness/).

The pre-trained models' [size and accuracy comparison](https://www.researchgate.net/figure/Comparison-of-Convolutional-Neural-Network-Architectures-in-Terms-of-Size-and-Performance_fig3_363738835)
![tuning.png](./utils/Comparison-of-Convolutional-Neural-Network-Architectures-in-Terms-of-Size-and-Performance.png)

The pre-trained models' [robustness and accuracy comparison](https://www.eetimes.eu/ai-tradeoff-accuracy-or-robustness/) 
![tuning.png](./utils/robustness.png)


### 3.1. The model structure & setup

- Pretrained `VGG16`
- Loss function: `CrossEntropyLoss()`
- Optimizer: `Adam()`
- Classifier layers were replaced

VGG16 Classifier Layers
```
 Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

Updated Classifier Layers
```
nn.Sequential(
    nn.Linear(num_features, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 516),
    nn.ReLU(inplace=True),
    nn.Linear(516, 4)
)
```

The training images were transformed:
```
transform_train = transforms.Compose([
        transforms.Resize(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.497,0.402,0.425), (0.308, 0.325, 0.301))
    ])
```

### 3.2. The following hyperparameters were tuned:
- `lr`: `ContinuousParameter(0.001, 0.1)`
- `batch-size`: `CategoricalParameter([32, 64])`

The model was trained for `30` epochs.

### 3.3. Best estimator & parameters

The completed hyperparameter tuning job is shown below.

![tuning.png](./Screenshots/Tuning complete.png)

| Best Estimator Parameters |
|---|
| - `lr`:0.03024 <br> - `batch-size`:32 |

### 3.4. Train the Best Estimater - Debugging and Profiling

The best model from the hyperparameter tuning was selected and the parameters are below.

The rules were defined as below.
```
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]
```

The following hooks were defined:

- `train.save_interval`: `1`
- `eval.save_interval`: `1`
- `predict.save_interval`: `1`


### 3.5. Results

The model trained successfully.

![model_train.png](./Screenshots/pre_model_train.png)

Although the `epoch` was set to 30, the validation loss started increasing at epoch 3 which trigger early stopping. The model's training and validation error plot is shown below.

![train_valid_error.png](./Screenshots/pre_model_train_error.png)

Based on the plot, the train and validation error both started decreasing and stabilized after few steps.

### 3.6. Model Deployment and Testing

The trained estimator was deployed with the following configuration.

- `initial_instance_count`=`1`, 
- `instance_type`=`"ml.m5.2xlarge"`

The endpoint was created without any problem. The inferences made from testing images which resulted in confusion matrix below.

|  | Pred_Mild | Pred_Mod | Pred_Non | Pred_VeryMild |
| ---|---|---|---|---|
| Mild |  0 | 0 |19 | 160 |
| Moderate |  0 | 0 | 2 | 10 |
| Non | 0 | 0 | 427 | 213 |
| VeryMild | 0 | 0 | 139 | 309 |

The accuracy score on testing data was `57.545%` which is below `80%` benchmark.

-----

## 4. CNN model

### 4.1. The model structure

- Loss function: `CrossEntropyLoss()`
- Optimizer: `Adam()`
- Model structure: the model consists of several blocks which was referenced from [here](https://www.kaggle.com/code/fazainsyahazka/1301184335-alzheimer-s-disease-classification-cnn).
`conv_block` is a block of convolutional layer with batch normalization and `dense_block` is dense linear layer with batch normalization.

The main model was defined as:

```
def net(num_classes, input_size, kernel_size, dropout_rate):
    
    model = nn.Sequential(
        nn.Conv2d(input_size, 128 , kernel_size=kernel_size, padding='same'),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=kernel_size, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        conv_block(input_size=64, kernel_size=kernel_size),
        conv_block(input_size=128, kernel_size=kernel_size),
        conv_block(input_size=192, kernel_size=kernel_size),
        nn.Dropout(dropout_rate),
        conv_block(input_size=256, kernel_size=kernel_size),
        nn.Dropout(dropout_rate),
        nn.Flatten(),
        dense_block(input_size=8000, output_unit=512, dropout_rate=0.7),
        dense_block(input_size=512, output_unit=128, dropout_rate=0.5),
        dense_block(input_size=128, output_unit=64, dropout_rate=0.3),
        nn.Linear(64, num_classes)
    )
    
    return model

```

### 4.2. The parameters and debugging setup
- `lr`: `0.0011`
- `batch-size`: `120`
- `epochs`: `100`

The following rules were defined:
```
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]
```

The following hooks were defined:

- `train.save_interval`: `1`
- `eval.save_interval`: `1`
- `predict.save_interval`: `1`

The training and validation metrics were monitored:

- `train:error`
- `validation:error`
- `train:accuracy'`
- `validation:accuracy`

The training images were updated. The `RandomHorizontalFlip` was removed and `ColorJitter` was added to adjust the color and the image size was changed to `176`.
```
transform_train = transforms.Compose([
        transforms.Resize((176,176)),
        transform.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.497,0.402,0.425), (0.308, 0.325, 0.301))
    ])
```

### 4.3. Results

The model trained successfully and the validation accuracy reached over `70%`.

![model_train.png](./Screenshots/model_train.png)


### 4.4. Model Deployment & Testing

The trained estimator was deployed with the following configuration.

- `initial_instance_count`=`1`, 
- `instance_type`=`"ml.m5.2xlarge"`

The endpoint was created without any problem.

![endpoint_inservice.png](./Screenshots/model_endpoint.png)

The model was tested on the test dataset and the following is the confusion matrix of the predictions.

|  | Pred_Mild | Pred_Mod | Pred_Non | Pred_VeryMild |
| ---|---|---|---|---|
| Mild |  83 | 0 |25 | 71 |
| Moderate |  2 | 3 | 2 | 5 |
| Non | 20 | 1 | 426 | 193 |
| VeryMild | 29 | 3 | 101 | 315 |

The accuracy score on testing data was `64.66%` which is an improvement from `VGG16` model.

----