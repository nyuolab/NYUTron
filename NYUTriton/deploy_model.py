#!/usr/bin/env python
# $ python deploy_model.py --model_name --model_type
# $ python deploy_model.py --model_name nyutron_readmission --model_type onnx

import os
import shutil
from shutil import copytree, ignore_patterns
import argparse
from jinja2 import Environment, FileSystemLoader
import json
from helpers import get_config

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name', type=str, default='nyutron_readmission', help='model name from our ModelHub')
parser.add_argument('--model_type', type=str, default='onnx', choices=['hf', 'onnx', 'trt'], help='model type/accelerator the way triton likes to name')


def build_directories(model_name, model_type):
    """
    This method constructs the underlying directory structure for the production model.

    Note that we ignore .bin files since we don't need to move the weights
    """
    print("LOADING {0} as {1} INTO NYUTRITON".format(model_name,model_type))
    
    #define directory paths
    src_model = "./hf_models/{0}".format(args.model_name)
    model_path = "./tritonmodelrepo/{0}_{1}".format(model_name,model_type)
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    
    copytree(
        src = src_model,
        dst = model_path+"/1/hf",ignore=ignore_patterns('*.bin'))
    
    #separately need to copy the base onnx model into another directory named model_name_base
    src_onnx = "./onnx_models/{0}/model.{1}".format(model_name,model_type)
    dst_base = "./tritonmodelrepo/{0}_base/1".format(model_name) 
    
    os.makedirs(dst_base, exist_ok=True)
    print("created folder : ", dst_base)
    
    shutil.copy(src = src_onnx,dst = dst_base)          
    print("created deployment dir at:{0}".format(dst_base))
    return model_path
 
    
def render_template(template_environment, template_filename, context):
    model_script = template_environment.get_template(
        template_filename).render(context)
    return model_script
 

def create_modelpyfile(model_name, model_type, model_path):
    """
    Creates the model's .py file that must be named model.py and specify the nVidia
    business logic for the underlying model that should include tokenizer, model, and 
    any other preprocessing steps.

    Args:
        model_name (str): model name
        model_type (str): model type must be hf, onnx, trt
        model_path (str): path to model directory

    Returns:
        None
    """
    #Extract info from HF config file
    model_cfg = get_config(model_name)

    print("Creating a {0} model.py".format(model_cfg['architectures'][0]))

    path = os.path.dirname(os.path.abspath(__file__))
    template_environment = Environment(
        autoescape=False,
        loader=FileSystemLoader(os.path.join(path, 'templates')),
        trim_blocks=False)
    
    fname = "{0}/1/model.py".format(model_path)
    context = {
        'model_name': model_name,
        'model_type': model_type,
        'model_architecture': model_cfg['architectures'][0],
        'input_name': model_cfg['input_name'],
        'output_name': model_cfg['output_name'],
        'max_seq_length': model_cfg["max_position_embeddings"]
    }
    
    with open(fname, 'w') as f:
        model_script = render_template(
            template_environment,
            'mdltemplate.py',
            context)
        f.write(model_script)


def create_modelcfgfile(model_name, model_type, model_path):
    """
    Creates the model's .pbtxt config file that must be named config.pbtxt and specify the nVidia
    triton config model.

    Args:
        model_name (str): model name
        model_type (str): model type must be hf, onnx, trt
        model_path (str): path to model directory

    Returns:
        None
    """
    #Extract info from HF config file
    model_cfg = get_config(model_name)

    print("Creating a {0} config.pbtxt".format(model_cfg['architectures'][0]))


    path = os.path.dirname(os.path.abspath(__file__))
    template_environment = Environment(
        autoescape=False,
        loader=FileSystemLoader(os.path.join(path, 'templates')),
        trim_blocks=False)
    
    fname = "{0}/config.pbtxt".format(model_path)
    context = {
        'model_name': '{0}_{1}'.format(model_name,model_type),
        'input_name': model_cfg['input_name'],
        'output_name': model_cfg['output_name'],
        'max_batch_size': model_cfg['max_batch_size']
    }
    
    with open(fname, 'w') as f:
        model_script = render_template(
            template_environment,
            'configtemplate.pbtxt',
            context)
        f.write(model_script)
 
 

########################################
 
if __name__ == "__main__":
    args = parser.parse_args()

    model_path = build_directories(
        model_name = args.model_name,
        model_type = args.model_type)
    
    create_modelpyfile(
        model_name = args.model_name,
        model_type = args.model_type,
        model_path = model_path)

    create_modelcfgfile(
        model_name = args.model_name,
        model_type = args.model_type,
        model_path = model_path)