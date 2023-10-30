import os
import logging
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from azure.storage.blob import ContainerClient, BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)

def download_asset_from_azure(
    
        azure_connection_string : str,
        container_name : str,
        base_path_blob : str,
        dest_dir_path : str,
        download_model : bool = False,
        file_ext : str = None,
        verbose : str = False
    ):
    """
    Download assests from azure.

    :param azure_connection_string: connection string to azure storage account
    :param container_name: container to which the data has to be downloaded
    :param base_path_blob: base_path of the blob which is to be replaced from the tree
    :param dest_dir_path: local directory where the blobs are to be downloaded
    :param file_ext: optional
    :return: download a list of files from azure container
    """
    try:
        if base_path_blob[-1] != '/':
            base_path_blob = base_path_blob + '/'
    except IndexError:
        pass

    container_client = ContainerClient.from_connection_string(conn_str=azure_connection_string,
                                                              container_name=container_name)
    generator = container_client.list_blobs()

    if file_ext is None:
        filepaths = [blob.name for blob in generator if blob.name.startswith(base_path_blob)]
    else:
        filepaths = [blob.name for blob in generator if blob.name.startswith(base_path_blob) and blob.name.endswith(file_ext)]
    if download_model:
        filepaths = [x for x in filepaths if 'exported_model' in x]
        model_path = os.path.join([x for x in filepaths if Path(x).suffix == '.pt'][0])
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    logging.log(logging.INFO, f"There are {len(filepaths)} files.")


    def download_obj(obj_file):

        file_temp = obj_file.replace(base_path_blob, "")
        file_temp = os.path.join(dest_dir_path, file_temp)
        dest_dir = os.path.split(file_temp)[0]
        os.makedirs(dest_dir, exist_ok=True )

        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=obj_file)
            with open(file_temp, "wb") as downloaded_file:
                downloaded_file.write(blob_client.download_blob().readall())
            return f"Successfully saved: {obj_file}"
        except ResourceNotFoundError as e:
            return f"Blob Not found: {obj_file}"
        except FileNotFoundError:
            return f"File Not found: {obj_file}"
        except Exception as e: # Ran out of space, or network issues,
            return f"Unknown Exception: {obj_file}, {e}"
    

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(download_obj, filepaths), total=len(filepaths)))
        if verbose:
            for r in results:
                logging.log(logging.INFO, r)
    if download_model:
        return model_path



def upload_asset_to_azure(
        azure_connection_string : str,
        container_name : str,
        asset_dir_path : str,
        file_name : str = None,
        exact : bool =False
    ):
    """

    :param azure_connection_string: connection string to azure storage account
    :param container_name: container to which the data has to be uploaded
    :param asset_dir_path:local directory from where the data needs to be transferred
    :param file_name:optional
    :param exact:optional
    :return:uploads a list of asset to azure container
    """
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    if file_name is None:
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(asset_dir_path) for f in filenames]
    else:
        if not exact:
            files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(asset_dir_path) for f in filenames if file_name in f]
        else:
            files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(asset_dir_path) for f in filenames if file_name == f]

    files = [x for x in files if ".gitkeep" not in x]
    for file in tqdm(files):
        blob_client = blob_service_client.get_blob_client(container=container_name,
                                                          blob=file)
        logging.log(logging.INFO, "uploading " + file.split("/")[-1] + '\n')
        with open(file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logging.log(logging.INFO, "Successfully uploaded " + file.split("/")[-1] + '\n')


