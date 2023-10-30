import os
import yaml
import torch
import argparse
from absl import app
from absl import logging
from pathlib import Path
from ml_collections import config_dict


from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from src.datahandler.cloud import download_asset_from_azure, upload_asset_to_azure
from src.datahandler.dataset import create_dataset
from src.engine.triplet import run as triplet_trainer
from src.engine.triplet import run_evaluation as tl_evaluation
from src.engine.logits import run as ce_trainer
from src.engine.logits import run_evaluation as ce_evaluation
# from src.logger import setup_logger
# setup_logger()

train_types = ["triplet", "logit", "all"]

CONNECTION_STR = "/subscriptions/bc52259d-226a-485d-9f7f-d54d673fe1c4/resourceGroups/learn-85e3c380-05da-4d0c-ac05-d288cbb8bf58/providers/Microsoft.Storage/storageAccounts/cloudshell801487437"
assert len(CONNECTION_STR) > 0, "Please provide value for Connection string."

logging.set_verbosity(logging.INFO)


def container_name(cfg, acc, train_type):
    """Container name to store the results in azure storage

    Args:
        cfg (configdict.ConfigDict): Training config
        acc (float or int): Accuracy of the model
        train_type (str): "triplet" or "logit"

    Returns:
        str : Container path
    """
    return os.path.join(
        cfg.data.container_name,
        cfg.data.container_results_path,
        train_type,
        cfg[train_type].model,
        f"val_acc_{acc}"
    )


def run_triplet(cfg, args):
    """Run Triplet training

    Args:
        cfg (configdict.ConfigDict): Config for training 
        args (argumentparser.ArgumentParser): Arguments
    """

    logging.log(logging.INFO, f"Training type: TRIPLET")

    if not args.only_evaluation:
        cfg.evaluation.exp_name = triplet_trainer(cfg.triplet, cfg.train, cfg.data)

    acc = tl_evaluation(cfg.evaluation, cfg.data)

    exp_path = os.path.join("out", Path(cfg.evaluation.exp_name).name)
    cont_name = container_name(cfg, acc, "triplet")
    
    if cfg.data.upload_azure:
        logging.log(logging.INFO, f"Triplet: Uploading results - {cont_name}")
        upload_asset_to_azure(
            azure_connection_string = CONNECTION_STR,
            container_name = cont_name,
            asset_dir_path = exp_path
        )



def run_logits(cfg, args):
    """Run Logits training

    Args:
        cfg (configdict.ConfigDict): Config for training 
        args (argumentparser.ArgumentParser): Arguments
    """

    logging.log(logging.INFO, f"Training type: LOGITS")

    if not args.only_evaluation:
        cfg.evaluation.exp_name = ce_trainer(cfg.logits, cfg.train, cfg.data)

    results = ce_evaluation(cfg.evaluation, cfg.data)
    acc = round(results['acc1'], 2)

    exp_path = os.path.join("out", Path(cfg.evaluation.exp_name).name)
    
    cont_name = container_name(cfg, acc, "logits")
    
    if cfg.data.upload_azure:
        logging.log(logging.INFO, f"Logits: Uploading results - {cont_name}")
        upload_asset_to_azure(
            azure_connection_string = CONNECTION_STR,
            container_name = cont_name,
            asset_dir_path = exp_path
        )


def parse_args():
    """Parse arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default='./conf/trainer.yaml')
    parser.add_argument("--only_evaluation", action='store_true')
    args = parser.parse_args()

    return args



def train(cfg, args):
    execute = []
    if cfg.train.type == "triplet":
        execute.append(run_triplet) #(cfg, args)
    elif cfg.train.type == "logits":
        execute.append(run_logits) #(cfg, args)
    elif cfg.train.type == "all":
        execute.append(run_logits) #(cfg, args)
        execute.append(run_triplet) #(cfg, args)
    else:
        ValueError("cfg.train.type should be one of the following values: \n ", train_types)
    
    for f in execute:
        f(cfg, args)


def run(argv):
    
    args = parse_args()

    with open(args.conf) as f:
        cfg= config_dict.ConfigDict(yaml.safe_load(f))

    # For config which has 
    for k in ["train", "evaluation"]:
        cfg[k].device = "gpu" if torch.cuda.is_available() else "cpu"
        logging.log(logging.INFO, f"{k} phase will use {cfg[k].device} Tensors")
        

    
    if not cfg.data.skip_azure: # Download the data from cloud
        logging.log(logging.INFO, f"Downloading data from CLOUD")
        download_asset_from_azure(
            CONNECTION_STR,
            cfg.data.container_name,
            cfg.data.base_path_img,
            cfg.data.local_dir_img,
            verbose = cfg.data.verbose
        )

    if cfg.data.create_dataset: # Create dataset pytorch dataset format
        logging.log(logging.INFO, f"Creating dataset..")
        create_dataset(
            cfg.data.local_dir_img,
            cfg.data.dataset_dir,
            split = (cfg.data.train_split, cfg.data.val_split, cfg.data.test_split)
        )
    

    train(cfg, args)


# def parse_config(argv):
#     args = parse_args()

#     with open(args.conf) as f:
#         cfg= config_dict.ConfigDict(yaml.safe_load(f))
    
#     if cfg.environment.azureml:
#         run_azure(cfg)
#     else:
#         from trainer import run



if __name__ == "__main__":
    app.run(run)

