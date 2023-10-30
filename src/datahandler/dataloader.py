import torch
from tqdm import tqdm
from torch.utils import data

from src.utils import *

class DataLoader(data.DataLoader):
    """
    Extend torch.data.DataLoader to utilize efficient dataloading with custom methods.
    """

    def __init__(self, dataset, batch_size, num_workers, shuffle, drop_last=False, generator=None, sampler=None):
        """_summary_

        Args:
            dataset (torch.data.dataset): Torch Dataset to be iterated over
            batch_size (int): Number of images per batch
            num_workers (int): Number of sub-processes to use for data loading
            shuffle (bool): Shuffle the data
            drop_last (bool, optional): Ignores the last batch (when the number of examples in dataset is not divisible by batch_size). Defaults to False.
            generator (torch.Generator, optional):  If not None, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers. Defaults to None.
            sampler (torch.data.Sampler or Iterable, optional): defines the strategy to draw samples from the dataset. Can be any Iterable with `__len__` implemented. If specified, shuffle must not be specified. Defaults to None.
        """
        super(DataLoader, self).__init__(
                dataset = dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                generator=generator,
                sampler=sampler
        )
        

    @torch.no_grad()
    def embedding_per_image(self, model, device):
        """
        Generate image embeddings of dataset with pandas.DataFrame as iterator from an embedding model.

        Args:
            model (torch.nn.Module): Embedding model
            device (str or torch.device): Representing the device on which a torch.Tensor and torch.nn.Module is or will be allocated.
        """

        model.eval()
        model.to(device)
        self.dataset.image_data["embedding"] = ""

        for sample in tqdm(self):
            anchors = sample['anchor'].to(device)
            embeds = model(anchors)
            indices = self.dataset.image_data[self.dataset.image_data['path'].isin(sample['name'])].index

            for image_name, embed, df_idx in zip(sample['name'], embeds, indices):
                assert image_name == self.dataset.image_data.iloc[df_idx]['path']
                self.dataset.image_data.iloc[df_idx]['embedding'] = embed.cpu().numpy()
        return