"""
@serve decorator
Original inspiration 
https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/torchserve.py
https://fastapi.tiangolo.com/
"""
from functools import wraps
from typing import List, Callable
import os
import logging 
import grequest #add this to requirements.txt
import inspect
from retrying import retry


logger = logging.getLogger()


def serve(func):
    """
    @serve is a decorator meant to be used on top of an inference function
    @serve
    def inference():
        perform_inference..

    What this will do is
    1. Override the base handler with a new handler with the inference function overloaded
    2. Create a new torchserve config 
    3. Archive a model.pt
    4. Start torchserve
    """
    @wraps(func)
    def serve_model(*args, **kwargs):
        
        handler_file = create_handler(func)

        config = create_torchserve_config(inference_http_port=7070, management_http_port=7071)
        
        model = archive_model(serialized_file="model.pt")

        start_torchserve(models=model, config=config)
        
        # Error handle and
        # Return some error code
        return 0

def create_handler(handler : Callable, handler_file : str = "handler.py"):
    """
    Create a new handler that inherits from the base handler
    Update the inference() function with the function captured by the decorator
    Write the handler to disk
    """
        inference_handler = inspect.getsource(handler)
        with open(handler_file, "w") as f:
            f.write("class CustomHandler(BaseHandler):\n")
            for index, line in enumerate(inference_handler):
                if index == 0:
                    f.write(f"\tdef inference(self, data, *args, **kwargs):")
                else:
                    f.write(f"\t{line}")
        return handler_file

def writetofile(filename : str = "model.py"):
    """
    @writetofile decorator to add on top of a torch.nn.Module to write it to disk as model.py
    """
    @wraps(cls)
    def archive_model(*args, **kwargs):
        lines = inspect.getsource(cls)
        with open(filename, "w") as f:
            f.write(lines)


# TODO: Create decorator for model archiver as well?
@serve
def handle_test():
    # Need to figure out how to pick up the handler and replace it by the inference function that exists in the base handler
    return NotImplemented


def start_torchserve(handler="base_handler", model_store="model_store", ts_config="config.properties", model_mar="model.mar"):
    """
    Wrapper on top of torchserve --start args
    """
    ts_config = create_torchserve_config()
    
    ts_command = f'torchserve --start'
    f'--model_store {model_store}'
    f'--ts_config_file {ts_config}'
    f'--models {model_mar}'
    f'--handler {handler}'

    logger.info(ts_command)

    # TODO: Add error handling
    os.system(ts_command)



def create_torchserve_config(inference_http_port : int = 8080, management_http_port : int = 8081, batch_size : int = 1, config_properties="config.properties"):
    """"
    Create a torchserve config.properties
    Currently this only supports inference and management port setting but 
    eventually should support everything in config manager
    """
    config = {
        f"inference_address": "http://0.0.0.0:{inference_http_port}",
        f"management_address": "http://0.0.0.0:{management_http_port}",
    }

    logger.info(config)

    for key, value in config.items():
        with open("config.properties", "w") as f:
            f.write(f"{key}={value}\n")
    
    return config_properties
    

def archive_model(serialized_file : str, model_file : str = None, model_name : str = "model", handler : str = "base_handler", version : int = 1, extra_files : List[str] = []):
    """
    wrapper on top of torch-model-archiver
    model_file is only needed for eager mode execution
    """

    # Need to hook this up to archive decorator as well that takes in the existing class and writes it to disk
    if model_file:
        arch_command = f"torch-model-archiver --model_file {model_file} --serialized_file {serialized_file} --model-name {model_name} --handler {handler} --version {version} --extra-files {extra_files}"
    else:
        arch_command = f"torch-model-archiver --serialized_file {serialized_file} --model-name {model_name} --handler {handler} --version {version} --extra-files {extra_files}"
    os.system(arch_command)
    
    return 0

@retry(stop_max_delay=1000)
def stop_torchserve():
    os.system("torchserve --stop")

def torchserve_request(endpoint : str= "", tasks : List[str] = [], extra_params : str = ""):
    """
    Make an inference or management request using async library
    request is synchronous so won't work well with dynamic batching and will be like setting batch size = 1 always
    Maybe don't need this function and can just make things clearer in documentation
    """
    rs = (grequest.get(task) for task in tasks)
    return [tasks, rs]
    