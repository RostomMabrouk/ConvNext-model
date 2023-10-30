import time
from collections import defaultdict, deque
import datetime
import math
import torch
import pandas as pd
import cv2
import os
import ast
import numpy as np
import torch.distributed as dist
from absl import logging
from PIL import Image as pil_image
from pathlib import Path
from src.datahandler.dataloader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]


def default_value():
    return None

def dataset_df(img_path, IMG_EXTENSIONS=["jpg", "png", "gif"]):
    """
    Create a dataframe that makes it easy to access specific type of files inside a folder. The dataframe must have fullpath to each image in one column and the parent folder name in class column.  

    Args:
        img_path (str or pathlib.PosixPath): Path to the images
        IMG_EXTENSIONS (list, optional): Image file extensions. Defaults to ["jpg", "png", "JPEG", "PNG", "JPG", "gif"].

    Returns:
        _type_: _description_
    """
    files = [f for ext in IMG_EXTENSIONS for f in Path(img_path).rglob(f'*.{ext}')]
    file_list = [(str(fullpath), fullpath.parent.name) for fullpath in files]
    df = pd.DataFrame(file_list, columns=["path", "class"])
    return df


def train_val_dataframes(dataset_path):
    """Fetches training and test data in pandas.DataFrame

    Args:
        dataset_path (str or pathlib.PosixPath): Path to dataset folder

    Returns:
        train_df (pandas.DataFrame) : Traning metadata in DataFrame
        val_df (pandas.DataFrame) : Validation metadata in DataFrame
    """
    train_df = dataset_df(img_path=Path(dataset_path, "train") )
    val_df = dataset_df(img_path=Path(dataset_path, "val") )
    return train_df, val_df


def cv2_image_loader(path):
    """Read an image in cv2 and convert it from BGR to RGB format

    Args:
        path (str): Provide image path

    Returns:
        image (numpy.ndarray): CV2 image in numpy.ndarray format.
    """
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def load_image(image_name):
    """Read an image in PIL format and return it in RGB format.

    Args:
        path (str or pathlib.PosixPath): Provide image path

    Returns:
        image (numpy.ndarray): CV2 image in numpy.ndarray format.
    """
    image = pil_image.open(image_name).convert('RGB')
    return image


def from_np_array(array_string):
    """Helper function to convert a numpy array in string format to numpy.ndarray format. 

    Args:
        array_string (str): String containing n dimensional array.

    Returns:
        numpy.array
    """
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


def save_embeddings(model, train_loader, val_loader, device, exp_name):
    """Generate and Save image embeddings 

    Args:
        model (torch.nn.Module): Model to generate image embeddings 
        train_loader (DataLoader): Training DataLoader to iterate over training images
        val_loader (DataLoader): Validation DataLoader to iterate over validation images
        device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        exp_name (str): Name of the experiment to save the generated embeddings
    """

    embed_transforms = val_loader.dataset.get_transform()

    trainds = train_loader.dataset
    train_loader = DataLoader(trainds, shuffle=False, batch_size=4, drop_last=False, num_workers=2)
    train_loader.dataset.set_transform(embed_transforms)
    train_loader.embedding_per_image(model, device)
    train_loader.dataset.image_data["path"] = train_loader.dataset.image_data["path"].apply(lambda x: x.split("/")[-1])
    img_embeds_path =os.path.join("outputs", exp_name, "image_embeddings.csv")
    train_loader.dataset.image_data.to_csv(img_embeds_path)
    logging.log(logging.INFO, f"Saved image emddings - {img_embeds_path}")

    
    val_loader.embedding_per_image(model, device )

    val_loader.dataset.image_data["path"] = val_loader.dataset.image_data["path"].apply(lambda x: x.split("/")[-1])
    val_img_embeds_path = os.path.join("outputs", exp_name, "val_image_embeddings.csv")
    val_loader.dataset.image_data.to_csv(val_img_embeds_path)
    logging.log(logging.INFO, f"Saved val image emddings - {val_img_embeds_path}")


def save_model(model, save_path):
    """Save torch.nn.Module

    Args:
        model (torch.nn.Module): PyTorch model to be save
        save_path (str or pathlib.PosixPath): Path to save the model
    """
    model.to('cpu')
    torch.save(model.state_dict(),  save_path)
    logging.log(logging.INFO, f"Successfully saved model at {save_path}")


def nearest_neighbour(image_embed_df, out_embedding, n_neighbours):
    """Fetch the classes of nearest neighbours of an embedding. 

    Args:
        image_embed_df (pandas.DataFrame) : DataFrame consisting of all the image embeddings for reference/
        out_embedding (numpy.array) : Image embedding 
        n_neighbours (int) : Number of nearest neighbours to be considered.

    Returns:
        top_classes (list) : List containing class names of nearest embeddings.
    """

    similarity = cosine_similarity(out_embedding, np.array(image_embed_df["embedding"].values.tolist()))[0]
    top_sims = similarity.argsort()[::-1]
    top_classes = []
    for sim in top_sims:
        c = image_embed_df.iloc[sim]['class']
        top_classes.append(c)
        if len(top_classes) == n_neighbours:
            break
    return top_classes


def is_dist_avail_and_initialized():
    """Check if the program is using distributed nodes and is initialized.

    Returns:
        bool : Return a bool value
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get the number of nodes used for training

    Returns:
        int : Returns number of nodes
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def cosine_scheduler(
            base_value,
            final_value,
            epochs,
            niter_per_ep,
            warmup_epochs=0,
            start_warmup_value=0,
            warmup_steps=-1
    ):
    """
    Set the cosine learning rate scheduler

    Args:
        base_value (float): Initial learning rate
        final_value (_type_): Last learning rate
        epochs (_type_): Number of epochs
        niter_per_ep (_type_): Number of iterations per epoch
        warmup_epochs (int, optional): Number of epochs where you to use a very low learning rate. Defaults to 0.
        start_warmup_value (int, optional): Define the "very low" learning rate to be used while training during warmup epochs. Defaults to 0.
        warmup_steps (int, optional): Number of warmup steps per epoch. Defaults to -1.

    Returns:
        np.ndarray: numpy.ndarray containing the learning rate to be used for every iteration.
    """
    
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    logging.log(logging.INFO, "Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule
    

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """Track and log metrics
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)


    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.log(logging.INFO, log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logging.log(logging.INFO, log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.log(logging.INFO, '{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))