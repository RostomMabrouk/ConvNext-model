import re
import shutil
import pathlib
import numpy as np
import pandas as pd
from absl import logging
from tqdm import tqdm
from typing import Union

create_dir = lambda x, y: pathlib.Path(x, y).mkdir(exist_ok=True, parents=True)
cls_extractor = lambda x: str(int(x.split("_")[4]))

def ndc_metadata_df(
        img_path : Union[str, pathlib.PosixPath],
        IMG_EXTENSIONS : Union[list, tuple] = ["jpg", "png", "JPEG", "PNG", "JPG", "gif"],
        look_parent : bool = True
    ):
    """
    Recursively find all files in a folder with custom extensions. Collect filenames, extension and class (parent folder) as a dataframe.

    Args:
        img_path (str or pathlib.PosixPath): Path to Folder (Ideally a folder with images)
        IMG_EXTENSIONS (list, optional): List/Tuple with all image extensions. Defaults to ["jpg", "png", "JPEG", "PNG", "JPG", "gif"].
        look_parent (bool, optional) : If True, class of a sample is set to name of the parent folder. 
        If set to False then the class extractor logic is used to set the classname of the samples (expects classname to be part of the filename).

    Returns:
        pd.DataFrame: Dataframe with filename, extension and class as columns
    """

    files = [f for ext in IMG_EXTENSIONS for f in pathlib.Path(img_path).rglob(f'*.{ext}')]
    if look_parent:
        file_list = [(re.sub('\.[A-Za-z]+','', str(fullpath.name)), fullpath.suffix, fullpath.parent.name) for fullpath in files]
        df = pd.DataFrame(file_list, columns=["filename", "extension", 'cls'])
    else:
        file_list = [ (re.sub('\.[A-Za-z]+','', str(fullpath.name)), fullpath.suffix) for fullpath in files]
        df = pd.DataFrame(file_list, columns = ["filename", "extension"])
        df['cls'] = df['filename'].apply(cls_extractor)
    return df


def create_dataset(
        source_dir : Union[str, pathlib.PosixPath],
        dest_dir : Union[str, pathlib.PosixPath],
        split : tuple =  (0.6, 0.2, 0.2) 
    ):
    """
    Creates a dataset folder with `for-training` and `holdout` folders from a data folder.

    Args:
        source_dir (str or pathlib.PosixPath): Folder container all the images
        dest_dir (str or pathlib.PosixPath): Create folder structure as following

        ```
            for-training
            ├
            |───── train
            │      ├── dog
            │      │     ├── 0.jpg
            │      │     ├── 1.jpg
            │      ├── cat
            │      │    ├── 10.jpg
            │      │    ├── 14.jpg
            │            ....
            |───── val
            │       ├── dog
            │       │    ├── 2.jpg
            |       |    ├── 3.jpg
            │       ├── cat
            │       │    ├── 12.jpg
            |       |    ├── 13.jpg
            │           ......
            holdout
            ├
            ├── dog
            │    ├── 234.jpg
            |    ├── 54645.jpg
            ├── cat
            │    ├── 254323.jpg
            |    ├── adfa.jpg
                ....
        ```
        split (tuple, optional): Train, val and holdout splits respectively. Defaults to (0.6, 0.2, 0.2).
    """

    data_df = ndc_metadata_df(source_dir)
    try:
        assert len(data_df)>0
    except AssertionError:
        logging.error(f"No images found in the source directory {source_dir}")
        raise

    training_dest_dir = pathlib.Path(dest_dir, "for-training")
    holdout_dest_dir = pathlib.Path(dest_dir, "holdout")
    training_dest_dir.mkdir(exist_ok=True, parents=True)
    holdout_dest_dir.mkdir(exist_ok=True, parents=True)
    train_split, val_split, test_split = split

    train_folder = pathlib.Path(training_dest_dir, "train")
    train_folder.mkdir(exist_ok=True, parents=True)
    val_folder = pathlib.Path(training_dest_dir, "val")
    val_folder.mkdir(exist_ok=True, parents=True)

    def cls_split(cls, cls_data_df):
        ldf = len(cls_data_df)
        train, validate, test = np.split( cls_data_df.sample(frac=1, random_state=24),  [ int(train_split*ldf), int( (train_split+test_split)*ldf) ] )
        
        create_files = { 
            train_folder : train,
            val_folder: validate,
            holdout_dest_dir: test 
        }

        for key, value in create_files.items():
            pathlib.Path(key, str(cls)).mkdir(exist_ok=True, parents=True)
            for idx, row in value.iterrows():
                src = pathlib.Path(source_dir, row.cls, f"{row.filename}{row.extension}")
                dst = pathlib.Path(key, row.cls, f"{row.filename}{row.extension}")
                shutil.copy(str(src), str(dst))
    
    logging.log(logging.INFO, f'Transferring images..')
    for cls in tqdm(data_df.cls.unique().tolist()):
        cls_data_df = data_df[data_df.cls == str(cls)]
        cls_split(cls, cls_data_df)
        logging.log(logging.DEBUG, f"{cls}", end=", ")
    print()

    return