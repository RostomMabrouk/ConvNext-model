import os
import numpy as np
import random
import torch
import json
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
from collections import defaultdict
from absl import logging
from torch.utils import data
from torchvision import transforms as tfs
from torch.utils import tensorboard as tb
from ml_collections import config_dict
from ast import literal_eval as le
from src.utils import default_value

from src.models import embed_model, freeze
from src.utils import *
from src.datahandler.dataloader import DataLoader
from src.datahandler.transformations import select_transforms

## Cache training images to speedup train dataloader
cache = {}


class TripletLossCosine(nn.Module):
    """Define Inverse of Cosine Similarity as Loss for three embeddings.
    """

    def __init__(self, margin):
        """ Create Triplet Loss with a margin
        Args:
            margin (int): Set the margin (minimum distance between anchor and negative sample)
        """
        super(TripletLossCosine, self).__init__()
        self.MARGIN =  margin
            
    def forward(self, anchor, positive, negative):
        dist_to_positive = 1 - F.cosine_similarity(anchor, positive)
        dist_to_negative = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(dist_to_positive - dist_to_negative + self.MARGIN)
        loss = loss.mean()
        return loss


class TripletSet(data.Dataset):
    """Generate a dataset that samples three images:
        - Anchor: Random instance from the dataset
        - Positive: Random instance from the dataset which shares the same class as Anchor.
        - Negative: Random instance from the dataset which doesn't shares the same class as Anchor.
    Note: Positive and Anchor are not the same sample.

    """

    def __init__(self, images_data, scope='train', transform=None):
        self.image_data = images_data
        self.scope = scope

        if transform:
            self.transform = transform
        else:
            logging.log(logging.DEBUG, "No transforms given, Default transforms will be used.")
            self.transform = tfs.Compose([
                tfs.Resize(size=(512,512)),
                tfs.CenterCrop(size=(256,256)),
                tfs.ToTensor(),
                tfs.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
            ])

        #logging.log(logging.INFO, 'Images: {}. Augmentation: {}. Scope: {}.'.format(len(self.image_data), transform, scope))
    
    def weighted_sampling(self):
        target = self.image_data['class'].values
        class_sample_count = self.image_data['class'].value_counts()
        weight = { k: 1. / v for (k,v) in class_sample_count.items()}
        samples_weight = np.array([weight[c] for c in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        return data.WeightedRandomSampler(samples_weight, len(samples_weight))

    def __getitem__(self, idx):

        '''
        For train and validation triplets are required, for prediction - only images;
        '''
        image_mode = 'RGB'
        row = self.image_data.iloc[idx]
        anchor_name = row['path']

        try:
            anchor = self.transform(get_image(anchor_name))
        except KeyError:    
            anchor = self.transform(image=cv2_image_loader(anchor_name))["image"]
            
        if self.scope == 'train' or self.scope == 'val' or self.scope=='validation':
            anchor_id = row['class']

            # import ipdb; ipdb.set_trace()
            positive_candidates = list(self.image_data[self.image_data['class'] == anchor_id]['path'])
            positive_candidates = [x for x in positive_candidates if x != anchor_name]

            if len(positive_candidates) == 0:
                positive_name = anchor_name
            else:
                positive_name = np.random.choice(positive_candidates)
            
            negative_candidate_class = random.choice([x for x in self.image_data['class'].unique() if x != anchor_id])
            #logging.log(logging.INFO, negative_candidate_class)
            negative_name = self.image_data[self.image_data["class"]==negative_candidate_class].sample()

            try:
                positive = self.transform(get_image(positive_name))
                negative = self.transform(get_image(negative_name["path"].values[0]))
            except KeyError: # Albumentations
                positive = self.transform(image=cv2_image_loader(positive_name))["image"]
                negative = self.transform(image=cv2_image_loader(negative_name["path"].values[0]))["image"]

            return {'name': anchor_name,
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                    }
        else:
            return {'name': anchor_name, 'anchor': anchor, 'class' : row['class']}

    def __len__(self):
        return len(self.image_data)
    
    def set_transform(self, transform):
        self.transform = transform

    def get_transform(self):
        return self.transform
    
    def create_dataloader(self, batch_size, shuffle, num_workers, drop_last):
        self.loader = data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def get_image(image_name):
    '''Returns cropped and resized image either from cache or from disk'''
    return cache.get(image_name) or load_image(image_name)
    

def triplet_loaders(model_config, dataset_path, num_workers):
    """Create train and val dataloaders for training in triplet setting

    Args:
        model_config (configdict.ConfigDict): Config with miscellaneous parameters
        dataset_path (str): Path to images 
        num_workers (int): Number of sub-processes to use for data loading

    Returns:
        torch.data.DataLoaders for traning and validation
    """

    train_df, val_df = train_val_dataframes(dataset_path)
    transforms = select_transforms(model_config)

    train_ds = TripletSet(train_df, scope="train", transform = transforms["train"])
    val_ds = TripletSet(val_df, scope="val", transform = transforms["val"])

    train_sampler = None
    shuffle = True # sampler option is mutually exclusive with shuffle
    if model_config['weighted_sampling']:
        train_sampler = train_ds.weighted_sampling()
        shuffle = False


    traindl = DataLoader(train_ds, batch_size=model_config["batch_size"], shuffle=shuffle, num_workers=num_workers, drop_last=False, sampler=train_sampler)
    valdl = DataLoader(val_ds, batch_size=model_config["batch_size"], shuffle=False, num_workers=num_workers, drop_last=False)

    return traindl, valdl 


@torch.no_grad()
def validate(model, validation_dataloader, device, loss_func): 
    """Run validation and return confusion matrix and class metrics. 

    Args:
        model (torch.nn.module): PyTorch module
        dataloader (torch.data.DataLoader): DataLoader to iterate over the validation data
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        loss_func (torch.nn.Module): Custom loss function to generate loss for one forward pass.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): Write logs to tensorboard. Defaults to None.

    Returns:
        dict : Returns dict with averages
    """

    model.eval()
    batch_losses = []

    for sample in validation_dataloader:
            for key in ['anchor','positive','negative']:
                sample[key] = sample[key].to(device)

            anchor_embed = model(sample['anchor'])
            positive_embed = model(sample['positive'])
            negative_embed = model(sample['negative'])
            loss = loss_func(anchor_embed, positive_embed, negative_embed) 

            batch_losses.append(loss.item())

    return batch_losses


def train_epoch(model, train_dataloader, optimizer, device, loss_func, current_iter_num=0, tb_writer=None):
    """Train one epoch over training dataloader

    Args:
        model (torch.nn.Module): Train the PyTorch module
        dataloader (torch.utils.data.DataLoader): DataLoader to iterate over the training dataset
        optimizer (torch.optimizer): Optimizer
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        loss_fn (torch.nn.Module): Loss function 
        tb_writer (torch.tensorboard.SummaryWriter): TensorBoard Writer for logging
        current_iter_num (int, optional): To track the loss at a specific iteration number. Defaults to 0.

    Returns:
        lte (int): Loss this epoch 
    """

    batch_losses = []
    for sample in tqdm(train_dataloader):
        model.train()
        optimizer.zero_grad()

        for key in ['anchor','positive','negative']:
            sample[key] = sample[key].to(device)

        anchor_embed = model(sample['anchor'])
        positive_embed = model(sample['positive'])
        negative_embed = model(sample['negative'])
        loss = loss_func(anchor_embed, positive_embed, negative_embed)  
        if tb_writer: 
            tb_writer.add_scalar("Iter_Loss/train", loss.item(), current_iter_num)
            current_iter_num += 1
        loss.backward()
        
        optimizer.step()

#         lr_scheduler.step()
        batch_losses.append(loss.item())
    return batch_losses


def fit(model, train_dataloader, validation_dataloader, optimizer, device, loss_func, EPOCHS, lr_scheduler=None,  tb_writer=None):
    """Fit the CNN model

    Args:
        model (torch.nn.Module): Train the PyTorch module
        train_dataloader (torch.utils.data.DataLoader): Training DataLoader to iterate over the training dataset
        val_loader (torch.utils.data.DataLoader): Validation DataLoader to iterate over the validation dataset
        optimizer (torch.optimizer): Optimizer
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        loss (torch.nn.Module): Loss function 
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler. Defaults to None.
        tb_writer (torch.tensorboard.SummaryWriter): TensorBoard Writer for logging

    Returns:
       model (torch.nn.Module): Returns trained model
    """

    train_losses = []
    validation_losses = []
    current_iter_num = 1
    # lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.4)
    for epoch in range(1, EPOCHS+1):
        batch_losses = train_epoch(model, train_dataloader, optimizer, device, loss_func, current_iter_num, tb_writer)
        current_iter_num += len(batch_losses)+1

        if lr_scheduler: lr_scheduler.step()

        mean_loss = np.mean(batch_losses)
    
        if tb_writer: tb_writer.add_scalar('Loss/train', mean_loss, epoch)        
        train_losses.append(mean_loss)
        val_loss = validate(model, validation_dataloader, device, loss_func)
        val_loss_mean = np.mean(val_loss)
        validation_losses.append(val_loss_mean)
        if tb_writer: tb_writer.add_scalar('Loss/validation', val_loss_mean, epoch)
        logging.log(logging.INFO, '====Epoch {}. Train loss: {}. Val loss: {}'.format(epoch,  mean_loss,  val_loss_mean))
    return model


@torch.no_grad()
def run_evaluation(evaluation_config, data_config):
    """Create model, load weights then run model on a given dataset.

    Args:
        cfg (configdict.ConfigDict): Config containing experiment name

    Raises:
        Exception: Raise exception if it failes to generate class metrics (if model is not able to predict all classes)
    """

    cluster_agg = ["nearest", "mode_neighbours"]
    global_avg = defaultdict(default_value)


    if os.path.isdir(evaluation_config.exp_name):
        pass
        # evaluation_config.exp_name = evaluation_config.exp_name
    elif os.path.isdir(os.path.join("outputs", evaluation_config.exp_name)):
        evaluation_config.exp_name = os.path.join("outputs", evaluation_config.exp_name)
    else:
        assert False, f"{evaluation_config.exp_name} doesn't exist."

    with open(os.path.join(evaluation_config.exp_name, 'model_config.json'), "r") as f:
        model_config = config_dict.ConfigDict(json.load(f))

    ds_path = os.path.join(data_config.dataset_dir, "holdout")
    img_df = dataset_df(img_path=ds_path)
    tfs = select_transforms(model_config)["val"]
    ds = TripletSet(img_df, scope="embed", transform=tfs )
    # loader= DataLoader(ds, batch_size=model_config.batch_size, shuffle=False, num_workers=evaluation_config.num_workers, drop_last=False)

    model = embed_model(model_type=model_config.model, embedSize = model_config.embedSize)
    model.load_state_dict(torch.load(os.path.join(evaluation_config.exp_name, f'{model_config.model}_final.pt')))
    model.eval()
    model.to(evaluation_config.device)

    img_embed_df = pd.read_csv(os.path.join(evaluation_config.exp_name, 'image_embeddings.csv'), converters={'embedding': from_np_array})
    classes_ = img_embed_df['class'].unique().tolist()

    for s in ["any_fault", "nearest", "fault_count", "mode_neighbours"]:
        ds.image_data[s] = None

    for idx, sample in tqdm(enumerate(ds)):
        sample['anchor'] = sample['anchor'].unsqueeze_(0).to(evaluation_config.device)
        embed = model(sample['anchor'])
        top_classes = nearest_neighbour(img_embed_df, embed.cpu().numpy(), n_neighbours=evaluation_config.neighbours)
        mode_neighbours = classes_.index(max(top_classes, key=top_classes.count)) 
        nearest = classes_.index(top_classes[0])
        fault_count = len(top_classes) - (top_classes.count('proper_tablet') + top_classes.count('proper'))
        any_fault = True if fault_count>0 else False
        ds.image_data.at[idx, "any_fault"] = any_fault
        ds.image_data.at[idx, "nearest"] =  classes_[nearest]
        ds.image_data.at[idx, "fault_count"] = fault_count
        ds.image_data.at[idx, "mode_neighbours"]= classes_[mode_neighbours]  
    
    with pd.ExcelWriter(os.path.join(evaluation_config.exp_name, 'evaluation.xlsx')) as writer:
        for i in [0.25, 0.5, 0.75]:
            ds.image_data[f'Fault-atleast-{i*100}%'] = ds.image_data.apply(lambda x: 'fault' if x.fault_count >= int(evaluation_config.neighbours * i) else "proper_tablet", axis=1)
            df_cross = pd.crosstab(ds.image_data[f'Fault-atleast-{i*100}%'], ds.image_data["class"])
            df_cross.to_excel(writer, sheet_name=f'ERROR_CROSTAB_ATLEAST{i*100}%')
        
        for s in cluster_agg:
            df_cross = pd.crosstab(ds.image_data[s], ds.image_data["class"])
            df_cross.to_excel(writer, sheet_name=f'ERROR_CROSTAB_{s.upper()}')
            global_avg[s] = round( (np.diag(df_cross).sum() / df_cross.to_numpy().sum())*100, 2)

        ds.image_data.to_excel(writer, sheet_name='Image_Data')

    del model
    torch.cuda.empty_cache()

    return global_avg['nearest']


#TODO: Profile (time it) this code
def run(model_config, train_config, data_config):

    # model_config = config["model"]
    # train_config = config["TRAIN"]
    exp_prefix = "TL"
    os.makedirs(os.path.join("outputs", "TL"))
    # train_config["exp_name"] = f"-{model_config['embedSize']}dim-{model_config['model']}-{model_config['freeze_layer']}_ep{model_config['epochs']}_aug{model_config['augment_type']}" \
    # f"_bs{model_config['batch_size']}_Wsample{int(model_config['weighted_sampling'])}_im{model_config['input_size']}"
    ds_path = os.path.join(data_config.dataset_dir, "for-training")
    train_config['exp_name'] = exp_prefix

    try:
        os.makedirs(f"outputs/{train_config['exp_name']}", exist_ok=False)
    except FileExistsError as e:
        # raise FileExistsError("File already exists, please choose another filename", e)
        pass
    except Exception as e:
        raise Exception (e)
    
    trainLoader, valLoader = triplet_loaders(model_config, dataset_path=ds_path, num_workers=train_config["num_workers"])
    
    # Get CNN
    model = embed_model(model_type=model_config["model"], embedSize = model_config["embedSize"], pretrained=model_config["pretrained"], freeze_at=model_config["freeze_layer"]) # TODO: Use different models
    model.to(train_config["device"])
    # if model_config["freeze_layer"]:
    #     model = freeze(model, model_config["freeze_layer"])

    loss_func = TripletLossCosine(margin = model_config["loss_margin"])
    
    # TODO: Custom optimizers
    optimizer = torch.optim.Adam(params=model.parameters(),lr= le(model_config.lr), weight_decay=le(model_config.weight_decay))
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, momentum=0.9)

    tb_writer = tb.SummaryWriter(os.path.join("outputs", train_config['exp_name'], "logs")) # Tensorboard writer
    
    # TODO: lr scheduler
    model = fit(model, trainLoader, valLoader, optimizer, train_config["device"], loss_func, model_config["epochs"], lr_scheduler=None, tb_writer = tb_writer)
    save_embeddings(model, trainLoader, valLoader, train_config['device'], train_config['exp_name'])
    
    # Save model
    save_path = os.path.join(os.getcwd(), os.path.join("outputs", train_config['exp_name'], f"{model_config['model']}_final.pt"))
    save_model(model, save_path)
    with open(os.path.join(os.getcwd(), "outputs", train_config["exp_name"], "model_config.json"), 'w', encoding='utf-8') as f:
        json.dump(model_config.to_dict(), f, ensure_ascii=False, indent=4)
    del model
    torch.cuda.empty_cache()
    return train_config['exp_name']
        