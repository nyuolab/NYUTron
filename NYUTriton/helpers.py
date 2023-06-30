from fileinput import filename
import json


def get_config(model_name):
    """
    Takes in the HF config.json and adds a few additional fields
    including the input and output names that we wiil use for the TRITON PIPELINE.
    Note that the HF model itself has different Input and Output names that are 
    wrapped into a logical statement in mdltemplate.py

    I'm also specifying the max_batch_size here.
    """
    src_model_cfg = "./hf_models/{0}/config.json".format(model_name)
    with open(src_model_cfg) as json_file:
        model_cfg = json.load(json_file)

    if model_cfg['architectures'][0] == 'BertForSequenceClassification':
        model_cfg['input_name'] = 'rawtext'
        model_cfg['output_name'] = 'logits'
        model_cfg['max_batch_size'] = 1

    return model_cfg


