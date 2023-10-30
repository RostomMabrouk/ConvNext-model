# Training NDC detection

> Changes: Add logger

Before training, please find config file in `conf/trainer.yaml`.

If you are downloading the data from the Azure, please mention `CONNECTION_STRING` in the `trainer.py` file.

All the results will be generated in experiment specific folder inside "out" folder. Please ensure you have created a out folder.

Generated results (based on evaluation folder):
 - Class metrics* : Logs class metrics such as True positives, false negatives etc. in one vs all format.
 - Also generates confusion matrix*.

* Will be able to generate only when model is able to predict every class at least once.

### Running the script

```bash
python main.py
```

### YAML Usage

#### Environment section

```yaml
environment:
    azureml : True  # set to True to run the experiment in azureml, False to run locally
    exp_name : ndc_detection # Name of the experiment
    cluster_name : cpu-cluster # Compute target to train the model in cluster name
    azure_config_path : ./conf/az_config.json # Path to Azure config to be able to run the model 
    vm_size : STANDARD_D2_v3 # Virtual machine size
    docker_base_img : mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04 # Docker container in the virtual machine
    register_model: True # Registering the model in the resourcegroup
    model_version: 0.1 # Model version
```    

If `azureml` key is set to `True` then then the script will run in **Azure Machine Learning studio**. 

Package also expects a `azure_config.json` for authentication purposes.


#### Data section

This part of the code assumes that data is to be downloaded from the 

```yaml
data:
  skip_azure: False # Skip downloading data from azure
  create_dataset : True # If you want to create dataset after downloading the dataset
  upload_azure : True # Upload validation results, models and logs to azure
  container_name : datadir # Name of the container
  base_path_img : upload/synmed_data #data/synmed_train_val
  local_dir_img : data/images # Path to save the images locally
  dataset_dir : data/ndc-name # Path to create a dataset with train and val as subdirectories
  eval_path : data/ndc-name/holdout # Evaluation path
  container_results_path: results/pytorch/exported_model # Where do you want to upload the results inside the container
  train_split : 0.64 # Train split
  val_split : 0.16 # val split
  test_split : 0.2 # holdout split
  verbose : True # Do you want to show the results while downloading the data
```

#### Train section

There are two training regimes:
    - Logits : uses cross-entropy loss
    - Triplet : Uses triplet loss

Three options in type:
    - logits
    - triplet
    - all (uses logits and triplet in this sequence)

For inference the constituents inside output folder must be preserved.
    - Weights file
    - Model config

```yaml
train :
    type: all # logits or triplet or all
    num_workers: 2 # Number of sub-processes to use for data loading
    device: cuda # CUDA or cpu
    finetune: null 
```

#### Evaluation section

Run evaluation on a custom dataset.

Requirements:
- Mention the path to evaluation folder in `eval_path` in `data` section.
- Folder must follow the eval-folder format mentioned below 
- You must locally save the trained model in the appropriate format mentioned in the training section for evaluation. 
- Mention the folder name in the exp_name key in order to independently predict on custom eval dataset. This folder must be ideally inside out folder.

```yaml
evaluation : 
    exp_name: null
    num_workers: 2
    device: cuda
    neighbours: 11
    batch_size: 1
    save_path: inference_results
```

#### Logits section

```yaml
logits :
    weighted_loss: False # Weighted loss 
    # weighted_loss: [1.0, 1.8] # Array example of weighted loss for binary classification
    weighted_sampling : False 
    model: resnet18 # resnet family, wideresnet and convnext models
    freeze_layer: null # Model is frozen till this layer
    epochs: 2 # Number of epocjs
    augment_type: 1 # Select augmentation type
    input_size: (224, 224) # Size of the image
    batch_size: 8 # Mini batch count
    pretrained: True # Use pretrained weights
    opt: adam # Use optimizer
    lr: 4.0e-4 # Learning rate
    weight_decay: 0.05 # Weighted decay
    weight_decay_end: null # Weight decay end
    layer_decay: 1.0 # Layer decay - specific to convnext famly models
    update_freq: 1 # Update frequency of weight decay on weights
    min_lr: 1.0e-6 # Least learning rate
    warmup_epochs: 0 # Warmup epochs (where learning rate is really low)
    warmup_steps: -1 # Warmup steps (learning rate is low till some iterations)
    trainable_bias_and_bn_params: False # Train bias and batch norm parameters
    log_every_iter: 100 # Log every N iterations while training an epoch
```

Introduced two keys to handle imblanced dataset:
- Weighted loss for weighted cross entropy: Two types of values
    - Bool (True): Default parameters are set based on the number of samples in each class
    - List: Array for custom parameters (not ideal for large number of classes)
- Weighted sampling:
    - Mini-batch sampling is based on the number of images based on their number of instances in the dataset.

Layer freezing:
- Must be model specific.
- If null (not None), then whole model is unfreezed for training.

#### Triplet section

Mention all the triplet parameters

```yaml
triplet:
    model: resnet18 # Options are resnet family and wideresnet50 and 101
    loss_margin: 0.6 # Minimum distance between positive and negative pairs
    lr: 1e-5 # Learning rate
    epochs: 3 # Number of epochs
    freeze_layer: layer2 # Name of the layer till the model needs to be freezed
    weight_decay: 1e-5 # Weight decay parameter
    embedSize: 128 # Embedding size of each image
    augment_type: 1 # Options: 0 1 2 2gray
    input_size: (224, 224) # Size of the each image size
    batch_size: 8 # Mini batch size
    pretrained: True # Use pretrained model
    weighted_sampling : True # Use weighted sampling
    log_every_iter: 100 # Log every N iterations while training an epoch
```

Loss margin ensures that there is minimum distance between positive and a negative embedding. This needs to be a balanced parameter.
 - If too low, model will have hard time distinguishing between classes.
 - If too high, training could be unstable because of large gradients. 


### Data Folder

#### Dataset Folder format

> Sample Image Data Loaders in PyTorch

```
for-training
├
|───── train
│      ├── daisy
│      │     ├── 0.jpg
│      │     ├── 1.jpg
│      │     ├── 4.jpg
│      ├── dandelion
│      │    ├── 10.jpg
│      │    ├── 11.jpg
│      │    ├── 14.jpg
│      ├── roses
│      ├── sunflowers
│      └── tulips
|───── val
│       ├── daisy
│       │    ├── 2.jpg
|       |    ├── 3.jpg
│       ├── dandelion
│       │    ├── 12.jpg
|       |    ├── 13.jpg
│       ├── roses
│       ├── sunflowers
│       └── tulips
```

#### Eval-folder format

> Consider 5 class problem. The folder format must follow the following format.

```
├── holdout
|   ├── daisy
│   │    ├── 18.jpg
│   │    ├── 51.jpg
|   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips

```


### Visualize results

```bash
tensorboard --logdir=out
```

### Model family

Please follow the naming convention followed below while using these:

#### ResNet family

- resnet18
- resnet34
- resnet50
- resnet101
- resnet152

#### WideResNet family

- wideresnet50
- wideresnet101

#### ConvNext family
> Works only for logits

- convnext_tiny
- convnext_small
- convnext_base
- convnext_large