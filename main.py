import os
import torch
import yaml
from absl import app
from ml_collections import config_dict

from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import Model
from azureml.core import Environment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import DockerConfiguration
from trainer import run, parse_args

def run_azure(cfg):
    """Run trainer.py in Azure Machine Learning Studio

    Args:
        cfg (configdict.ConfigDict): Training configuration
    """
    assert os.path.isfile(cfg.environment.azure_config_path), "Error while processing azure config file."
    ws = Workspace.from_config(cfg.environment.azure_config_path)
    cluster_name = cfg.environment.cluster_name

    # Verify that cluster does not exist already
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size=cfg.environment.vm_size, 
                                                               max_nodes=1)

        # Create the cluster with the specified name and configuration
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

        # Wait for the cluster to complete, show the output log
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)


    pytorch_env = Environment.from_conda_specification(name='torch-1.11-cpu', file_path='./conda-deps.yml')

    # Specify a GPU base image
    #pytorch_env.docker.enabled = True
    docker_config = DockerConfiguration(use_docker=True)
    pytorch_env.docker.base_image = cfg.environment.docker_base_img
    
    project_folder = os.getcwd()
    
    src = ScriptRunConfig(source_directory=project_folder,
                          script='trainer.py',
                          compute_target=compute_target,
                          environment=pytorch_env,
                          docker_runtime_config=docker_config)
    
    run_exp = Experiment(ws, name=cfg.environment.exp_name).submit(src)
    run_exp.wait_for_completion(show_output=True)

    # TODO: Download files from jobs
    model_names = {}
    if cfg.train.type == "logits":
        model_names["CE"] = cfg.logits.model
    elif cfg.train.type == "triplet":
        model_names["TL"] = cfg.triplet.model
    else:
        model_names["TL"] = cfg.triplet.model
        model_names["CE"] = cfg.logits.model
    
    for key, val in model_names.items():
        # model_path = os.path.join(os.getcwd(), 'out', key) 
        local_save_path = os.path.join('outputs', 'azure')
        # Create a model folder in the current directory
        os.makedirs(local_save_path, exist_ok=True)
        
        model_suffix = f'{key}_v{cfg.environment.model_version}'
        if cfg.environment.register_model:
            model =  run_exp.register_model(
                model_name=f'event-det-{model_suffix}',  # Local file to upload and register as a model.
                model_framework=Model.Framework.PYTORCH,  # Framework used to create the model.
                model_framework_version=torch.__version__,  
                description='EVENT classification model in PyTorch',
                tags={
                  'area':'pill-event',
                  'type':'image classification',
                  'framework':'torch'
                  },
                model_path=f'outputs/{key}',
                # version = float(cfg.environment.model_version)
            )
            model.download(target_dir=os.path.join(local_save_path, model_suffix), exist_ok=True)
        else:
            run_exp.download_file(
                name=f'outputs/{key}/{val}_final.pt',
                output_file_path=f'./outputs/{model_suffix}/{val}_final.pt') 
            run_exp.download_file(
                name=f'outputs/{key}/model_config.json',
                output_file_path=f'./outputs/{model_suffix}/model_config.json')



def main():
    args = parse_args()

    with open(args.conf) as f:
        cfg= config_dict.ConfigDict(yaml.safe_load(f))
    
    if cfg.environment.azureml:
        run_azure(cfg)
    else:
        app.run(run)


if __name__ == "__main__":
    main()
