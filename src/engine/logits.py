import os
import json
#from typing import OrderedDict
import timm
import torch
import collections
import numpy as np
import seaborn as sns
import torch.nn as nn
from absl import logging
from tqdm import tqdm
from typing import Union, Optional
from torch.utils import data
from timm.utils import accuracy
from matplotlib import pyplot as plt
from ml_collections import config_dict
from torchvision import transforms as tfs
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.logging import logging_redirect_tqdm

from sklearn.utils import class_weight
from src.models import load_resnet
from src.utils import *
from src.optim_factory import create_optimizer, LayerDecayValueAssigner
from src.datahandler.transformations import select_transforms

## Cache training images to speedup train dataloader
cache = {}
ITERATIONS  = 0
# LOG = logging.getLogger(__name__)

def get_image(image_name):
    '''Returns cropped and resized image either from cache or from disk'''
    return cache.get(image_name) or load_image(image_name)


class CELoss(nn.Module):
    """Create Cross Entropy loss with/without weights.
    """

    def __init__(self, ndc_weight=None):

        super(CELoss, self).__init__()        
        if ndc_weight is not None:
            self.ndc_criterion = nn.CrossEntropyLoss(weight = ndc_weight, reduction='mean')
        else:
            self.ndc_criterion = nn.CrossEntropyLoss()
    
    def forward(self, ndc_out, ndc_target):#, ndc_out, ndc_target):
        ndc_loss = self.ndc_criterion(ndc_out, ndc_target)
        return {
            'ndc': ndc_loss,
        }


class CustomDataset(data.Dataset):
    """torch.data.Dataset to iterate over images and return image tensor and label.
    """

    def __init__(self, images_data, transform=None):
        self.image_data = images_data
        # self.ndc_classes = images_data['class'].unique()
        self.ndc_classes = collections.OrderedDict(
            sorted(
                { val : i for i, val in enumerate(sorted(images_data['class'].unique().tolist()))}.items()
            )
        )

        if transform:
            self.transform = transform
        else:
            logging.log(logging.WARNING, "No transforms given, Default transforms will be used.")
            self.transform = tfs.Compose([
                tfs.CenterCrop(size=(128,128)),
                tfs.ToTensor(),
                tfs.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
            ])


    def weighted_sampling(self):
        """Create weighted sampling to tackle class imbalance problem.
        """
        target = self.image_data['class'].values
        class_sample_count = self.image_data['class'].value_counts()
        weight = { k: 1. / v for (k,v) in class_sample_count.items()}
        samples_weight = np.array([weight[c] for c in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        return data.WeightedRandomSampler(samples_weight, len(samples_weight))


    def __getitem__(self, idx):
        '''
        Sample image from the iterator.
        '''
        image_mode = 'RGB'
        row = self.image_data.iloc[idx]
        anchor_name = row['path']

        try:
            anchor = self.transform(get_image(anchor_name))
        except KeyError:    
            anchor = self.transform(image=cv2_image_loader(anchor_name))["image"]
        return {'image':anchor , 'class': self.ndc_classes[row['class']]}

    def __len__(self):
        return len(self.image_data)
    
    def set_transform(self, transform):
        self.transform = transform

    def get_transform(self):
        return self.transform
    
    def create_dataloader(self, batch_size, shuffle, num_workers, drop_last):
        self.loader = data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def dataloader(model_config, im_path , num_workers):
    """Create train and val dataloaders for training

    Args:
        model_config (configdict.ConfigDict): Config with miscellaneous parameters
        im_path (str): Path to images 
        num_workers (int): Number of sub-processes to use for data loading

    Returns:
        torch.data.DataLoaders for traning and validation
    """

    transformations = select_transforms(model_config)

    train_image_data = dataset_df(os.path.join(im_path, "train"))
    train_ds = CustomDataset(train_image_data, transform=transformations["train"])

    train_sampler = None
    shuffle = True # sampler option is mutually exclusive with shuffle
    if model_config['weighted_sampling']:
        train_sampler = train_ds.weighted_sampling()
        shuffle = False

    train_dl = data.DataLoader(train_ds, shuffle=shuffle, batch_size=model_config['batch_size'], num_workers=num_workers, sampler=train_sampler)

    val_image_data = dataset_df(os.path.join(im_path, "val"))
    val_ds = CustomDataset(val_image_data, transform=transformations['val'])
    val_dl = data.DataLoader(val_ds, shuffle=False, batch_size=model_config['batch_size']//2, num_workers=num_workers)

    # import ipdb; ipdb.set_trace()
    assert train_ds.ndc_classes == val_ds.ndc_classes, "Number of train and validation ndc classes are not same!"
    return train_dl, val_dl


@torch.no_grad()
def validate(model, dataloader, device, tb_writer=None):
    """Run validation and return confusion matrix and class metrics. 

    Args:
        model (torch.nn.module): PyTorch module
        dataloader (torch.data.DataLoader): DataLoader to iterate over the data
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): Write logs to tensorboard. Defaults to None.

    Returns:
        dict : Returns dict with averages
    """

    model.eval()
    model.to(device)
    
    metric_logger = MetricLogger(delimiter="  ")
    criterion = torch.nn.CrossEntropyLoss()
    
    prediction= []
    actual = []
    classes = {value: key for key,value in dataloader.dataset.ndc_classes.items() }
    map_cls = lambda x: classes[x.cpu().item()]
    top_k = min(len(classes)-1, 5)

    for sample in tqdm(dataloader):
        out= model(sample["image"].to(device))
        loss = criterion(out, sample['class'].to(device))
        
        acc1, acc_k = accuracy(out, sample['class'].to(device), topk=(1, top_k))
        batch_size =  sample['image'].shape[0]
        
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters[f'acc{top_k}'].update(acc_k.item(), n=batch_size)
        
        _, pred_batch = torch.max(out.cpu(), 1)
        prediction.extend(list(map(map_cls, pred_batch)))
        actual.extend(list(map(map_cls, sample['class'])))
        
    logging.log(logging.INFO, '* Acc@1 {top1.global_avg:.3f} Acc@{top_k} {topk.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top_k=top_k, topk=metric_logger.meters[f'acc{top_k}'], losses=metric_logger.loss))

    global_avg = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    try:
        # Confusion matrix
        results_df= pd.DataFrame(np.vstack([prediction, actual]).T ,columns=['predicted_class','actual_class'])
        confusion_matrix = pd.crosstab(results_df['actual_class'], results_df['predicted_class'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure(figsize = (18,18), dpi=120)
        cm_fig = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d').get_figure()
        plt.close(cm_fig)
        cm_fig.patch.set_facecolor('white')
        global_avg['confusion_matrix'] = cm_fig
        
        # Class metrics
        # TODO: Handle cases when few classes are not predicted.
        if len(results_df.actual_class.unique()) == len(results_df.predicted_class.unique()):
            cls_metrics = { 'class': [], 'count': [], 'TP': [], 'FP': [], 'FN': [] }
            for idx, row in confusion_matrix.iterrows():
                cls_metrics['class'].append(idx)
                cls_metrics['count'].append(row.sum())
                cls_metrics['TP'].append(row[idx])
                cls_metrics['FP'].append(confusion_matrix[idx].sum() - row[idx])
                cls_metrics['FN'].append(row.sum() - row[idx])
            df_cls = pd.DataFrame(cls_metrics)
            global_avg['class_metrics'] = df_cls

    except:
        logging.log(logging.WARNING, 'Error creating confusion matrix or class metrics.')

    return global_avg


def run_evaluation(cfg, data_cfg):
    """Create model, load weights and run model on a given dataset.

    Args:
        cfg (configdict.ConfigDict): Config containing experiment name

    Raises:
        Exception: Raise exception if it failes to generate class metrics (if model is not able to predict all classes)
    """
    
    # cfg.exp_name = f'./outputs/{cfg.exp_name}'
    # assert os.path.isdir(cfg.exp_name) or os.path.isdir(os.path.path("outputs", cfg.exp_name))
    if os.path.isdir(cfg.exp_name):
        exp_path = cfg.exp_name
    elif os.path.isdir(os.path.join("outputs", cfg.exp_name)):
        exp_path = os.path.join("outputs", cfg.exp_name)
    else:
        assert False, f"{cfg.exp_name} doesn't exist."

    # import ipdb; ipdb.set_trace()
    with open(os.path.join(exp_path, 'model_config.json'), "r") as f:
        model_config = config_dict.ConfigDict(json.load(f))

    transformations = select_transforms(model_config)
    
    ds_path = os.path.join(data_cfg.dataset_dir, "holdout")
    if not os.path.isdir(ds_path):
        ds_path = data_cfg.eval_path
    logging.log(logging.INFO, f"Eval path: {ds_path}")
    val_image_data = dataset_df(ds_path)
    logging.log(logging.INFO, f"Total number of images: {len(val_image_data)}")

    val_ds = CustomDataset(val_image_data, transform=transformations['val'])
    valdl = data.DataLoader(val_ds, shuffle=False, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], drop_last=False)

    logging.log(logging.INFO, f"Number of images for evaluation: {len(val_ds)}")
    
    if 'res' in model_config.model:
        model = load_resnet(model_config.num_classes, create_fc=True, model_type=model_config['model'], pretrained=False, backbone_freeze=None)
    elif 'convnext' in model_config.model:
        model = timm.create_model(model_config.model, num_classes=model_config.num_classes, pretrained=False)
    else:
        raise Exception('Unknown model type')

    load_path = os.path.join(exp_path, f"{model_config.model}_final.pt")
    model.load_state_dict(torch.load(load_path))

    stats = validate(model, valdl, cfg['device'])
    # import ipdb; ipdb.set_trace()
    if cfg.save_path is None:
        cfg.save_path = "./"
    else:
        os.makedirs(cfg.save_path, exist_ok=True)

    cfm = stats.pop('confusion_matrix')
    cfm.suptitle(f'{model_config.model}', fontsize=40)
    cfm.savefig(os.path.join(cfg.save_path, f"{model_config.model}_confusion_matrix.png"))

    try:
        stats['class_metrics'].to_csv(os.path.join(cfg.save_path, f"{model_config.model}_class_metrics.csv"))
        stats.pop('class_metrics')
    except KeyError:
        logging.log(logging.WARNING, 'Could not generate class metrics')
        pass

    return stats


def acc(output, target):
    """Calculate accuracy

    Args:
        output (torch.Tensor): Output from the model.
        target (torch.Tensor): Expected target

    Returns:
        correct (torch.Tensor)
    """

    _, predicted = torch.max(output, 1)
    # count = target.size(0)
    correct = (target == predicted).sum().item()
    # train_acc = (100 * correct_train) / target_count
    return correct


def train_epoch(
    model : nn.Module,
    loss_fn : nn.Module,
    optimizer : torch.optim,
    dataloader : torch.utils.data.DataLoader,
    device : Union[str, torch.device] = 'cuda',
    tb_writer : torch.utils.tensorboard.SummaryWriter = None,
    lr_schedule_values : np.ndarray = None,
    wd_schedule_values : np.ndarray = None,
    update_freq : Optional[int] = None,
    start_steps : Optional[int] = None,
    num_training_steps_per_epoch : Optional[int] = None,
    log_every_iter : Optional[int] = 100
 ):
    """Train one epoch over training dataloader

    Args:
        model (torch.nn.Module): Train the PyTorch module
        loss_fn (torch.nn.Module): Loss function 
        optimizer (torch.optimizer): Optimizer
        dataloader (torch.utils.data.DataLoader): DataLoader to iterate over the training dataset
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        tb_writer (torch.tensorboard.SummaryWriter): TensorBoard Writer for logging
        lr_schedule_values (np.ndarray, optional): Learning rate values for the epoch. Defaults to None.
        wd_schedule_values (np.ndarray, optional): weight decay values for the epoch. Defaults to None.
        update_freq (int, optional): Update frequency. Defaults to None.
        start_steps (int, optional): Start steps. Defaults to None.
        num_training_steps_per_epoch (int, optional): Number of training steps per epoch. Defaults to None.
        log_every_ite (int, Optional) : Log every 'n' number of iterations. Default to 100.

    Returns:
        model (torch.nn.Module): Trained model on epoch
        lte (int): Mean Loss this epoch
        ndc_acc (int): Accuracy this epoch
    """
    
    global ITERATIONS
    model.train()
    model.to(device)
    lte = [] # loss this epoch
    ndc_correct = 0
    count = 0
    # with logging_redirect_tqdm():
    for iter_step, sample in enumerate(tqdm(dataloader)):
        ITERATIONS = ITERATIONS+1
        step = iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        try:
            if lr_schedule_values is not None or wd_schedule_values is not None and ITERATIONS % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]
        except KeyError:
            pass

        optimizer.zero_grad()
        ndc_out = model(sample["image"].to(device))

        ndc_correct += acc(ndc_out.cpu(), target = sample['class'].cpu())
        # ndc_correct += acc(ndc_out.cpu(), target = sample['ndc'].cpu())
        count += sample['image'].size(0)

        loss_dict = loss_fn(ndc_out.cpu(), sample["class"]) #ndc_out.cpu(), sample["ndc"])
        loss = loss_dict['ndc'] #+ loss_dict['ndc']
        if tb_writer:
            # tb_writer.add_scalar('loss/ndc', loss_dict['ndc'], ITERATIONS)
            tb_writer.add_scalar('loss/iter', loss.item(), ITERATIONS )
        
        if ITERATIONS % log_every_iter == 0:
            # TODO: Set the debug level
            logging.log(logging.INFO, f"[ITER: {ITERATIONS}] Loss : {loss.item():.3f}")

        lte.append(loss)
        loss.backward()
        optimizer.step()

    ndc_acc = 100*(ndc_correct)/count

    lte = sum(lte)/len(lte)

    del sample
    torch.cuda.empty_cache()
    
    
    return model, lte, ndc_acc


def fit(args, model, loss, optimizer, train_dataloader, val_loader, device, tb_writer):
    """Fit the CNN model

    Args:
        args (configdict.ConfigDict): Hyperparameters for training
        model (torch.nn.Module): Train the PyTorch module
        loss (torch.nn.Module): Loss function 
        optimizer (torch.optimizer): Optimizer
        train_dataloader (torch.utils.data.DataLoader): Training DataLoader to iterate over the training dataset
        val_loader (torch.utils.data.DataLoader): Validation DataLoader to iterate over the validation dataset
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        tb_writer (torch.tensorboard.SummaryWriter): TensorBoard Writer for logging

    Returns:
       model (torch.nn.Module): Returns trained model
    """


    total_batch_size = train_dataloader.batch_size * args.update_freq * get_world_size()
    num_training_steps_per_epoch = len(train_dataloader.dataset) // total_batch_size

    lr_schedule_values  = None
    if abs(args.lr - args.min_lr) > 1e-8:
        logging.log(logging.INFO, "Using Cosine LR scheduler")
        lr_schedule_values = cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    logging.log(logging.INFO, "Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    for i_epoch in range(args.epochs):
        model, lte, train_acc = train_epoch (model, loss, optimizer, train_dataloader, device=device,
        tb_writer=tb_writer, lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, update_freq=args.update_freq,
        start_steps=i_epoch*num_training_steps_per_epoch, num_training_steps_per_epoch=num_training_steps_per_epoch, log_every_iter=args.log_every_iter)
        logging.log(logging.INFO, f"--- Loss at Epoch {i_epoch} : {lte}")
        test_stats = validate(model, val_loader, device, tb_writer)

        if tb_writer:
            tb_writer.add_scalar('Loss/Epoch', lte, i_epoch)
            tb_writer.add_scalar('Accuracy/ndc', train_acc, i_epoch)
            try:
                cfm = test_stats.pop('confusion_matrix')
                tb_writer.add_figure("perf/Confusion matrix", cfm, i_epoch)
                cfm.savefig(os.path.join(tb_writer.log_dir, "confusion_matrix.png"))
                path_=r'D:\Finarb\Azure_name_classification_V1\ndc-det-v0.2.1\outputs\CE'
                test_stats['class_metrics'].to_csv(os.path.join(tb_writer.log_dir, "class_metrics.csv"))
                test_stats.pop('class_metrics')
            except KeyError:
                logging.log(logging.WARNING, 'KeyError: While saving confusion matrix and/or class_metrics')
                pass
            
            for key, val in test_stats.items():
                tb_writer.add_scalar(f'perf/{key}', val, i_epoch)
        print()
    return model


def run(model_config, train_config, data_config):
    """Run training

    Args:
        model_config (configdict.ConfigDict): Model Parameters to create the model
        train_config (configdict.ConfigDict): Training parameters

    Raises:
        Exception: Unknowm model type

    Returns:
        exp_name (str) : Name of the folder where logs and weights are stored 
    """

    exp_prefix = 'CE'
    # train_config["exp_name"] = f"{model_config['model']}-{model_config['freeze_layer']}_ep{model_config['epochs']}_aug{model_config['augment_type']}" \
    # f"_bs{model_config['batch_size']}_Wsample{int(model_config['weighted_sampling'])}_im{model_config['input_size']}"
    ds_path = os.path.join(data_config.dataset_dir, "for-training")

    traindl, valdl = dataloader(model_config, ds_path, num_workers = train_config['num_workers'])
    
    #TODO This can be a dataset function
    if isinstance(model_config["weighted_loss"], list):
        # exp_prefix = exp_prefix + '-(custW)-'
        loss = CELoss(ndc_weight = torch.Tensor(model_config["weighted_loss"]))
    elif model_config['weighted_loss']:
        exp_prefix = exp_prefix + '-(Wloss)-1-'
        ndc_class_weights= class_weight.compute_class_weight( 
                                        class_weight = 'balanced',
                                        classes = traindl.dataset.image_data['class'].unique(),
                                        y = traindl.dataset.image_data['class']
                                    )
        ndc_class_weights=torch.tensor(ndc_class_weights,dtype=torch.float)

        loss = CELoss(ndc_weight = ndc_class_weights)
    else:
        # exp_prefix = exp_prefix + '-Wloss-0-'
        loss = CELoss(ndc_weight = None)

    # train_config['exp_name'] = exp_prefix + train_config['exp_name']
    train_config['exp_name'] = exp_prefix


    model_config.num_classes = len(traindl.dataset.ndc_classes)

    if 'res' in model_config.model:
        model = load_resnet(model_config.num_classes, create_fc=True, model_type=model_config.model, pretrained=model_config['pretrained'], backbone_freeze=model_config['freeze_layer'])
    elif 'convnext' in model_config.model:
        model = timm.create_model(model_config.model, num_classes=model_config.num_classes, pretrained=model_config.pretrained)
    else:
        raise Exception('Unknown model type')

    # optimizer = torch.optim.Adam(params=model.parameters(),lr=le(model_config.lr), weight_decay=le(model_config.weight_decay))
    if  model_config.layer_decay < 1.0 or model_config.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert model_config.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(model_config.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        logging.log(logging.INFO, "Assigned values = %s" % str(assigner.values))

    optimizer = create_optimizer(
        args=model_config, model=model, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    

    os.makedirs(os.path.join("outputs", "CE"), exist_ok=True)
    tb_writer = SummaryWriter(os.path.join("outputs", f"{train_config['exp_name']}", "logs"))
    model = fit(model_config, model, loss, optimizer, traindl, valdl, train_config['device'], tb_writer )

    save_path = os.path.join(os.getcwd(), "outputs", f"{train_config['exp_name']}", f"{model_config['model']}_final.pt")
    save_model(model, save_path)
    
    with open( os.path.join(os.getcwd(), "outputs", f'{train_config["exp_name"]}', "cls_dict.json"), 'w') as f:
        json.dump(traindl.dataset.ndc_classes, f)

    with open(os.path.join(os.getcwd(), 'outputs', train_config["exp_name"] , "model_config.json"), 'w', encoding='utf-8') as f:
        json.dump(model_config.to_dict(), f, ensure_ascii=False, indent=4)
    
    logging.log(logging.INFO, os.listdir(os.path.join('outputs', 'CE')))

    del model
    torch.cuda.empty_cache()

    return train_config.exp_name
#%%