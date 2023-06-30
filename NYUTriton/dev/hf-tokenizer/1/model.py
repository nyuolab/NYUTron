from nemo.collections import nlp as nemo_nlp
import numpy as np
import torch
import json
import triton_python_backend_utils as pb_utils
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from transformers import AutoTokenizer
from torch.utils.dlpack import from_dlpack

def postprocessing(results, labels):
    return [labels[str(r)] for r in results]


def zero_pad(inp, dim1=192):
    return np.pad(inp, ((0,0),(0, dim1-inp.shape[1])))

class TritonPythonModel:
    
    def __init__(self):
        ## TOIMPROVE performance later
        # self.dl = create_infer_dataloader
        self.tokenizer = AutoTokenizer.from_pretrained("/models/hf-inference/1/nyutron_test")

        self.s1_input_names = ['rawtext',]
        self.s1_output_names = ['textout',]


    def execute(self, requests, verbose=True):
      
      responses = []

      for request in requests:

        # get the text payload
        raw_strings = pb_utils.get_input_tensor_by_name(request, self.s1_input_names[0]).as_numpy()
        text = raw_strings.item().decode()
        
        # model_mode = pb_utils.get_input_tensor_by_name(request, "model_mode").as_numpy()[0, 0]
        model_mode = 0

        # call the tokenizer on the server side
        encoded_input = self.tokenizer(text)
        encoded_input['input_ids'] = zero_pad(encoded_input['input_ids'])
        encoded_input['attention_mask'] = zero_pad(encoded_input['attention_mask'])
        encoded_input['token_type_ids'] = zero_pad(encoded_input['token_type_ids'])

        # list of supported models
        model_dict = {0: 'model_onnx',        # ONNX as is
                      1: 'model_trt',    # TensorRT plan from ONNX tuned
        }    

        # conditional cast to INT32
        def cast(data, model_mode):
          """
          cast conditionally to INT32 since TRT does not like INT64
          """

          if model_dict[model_mode] == 'model_trt':
            return data.astype(np.int32)
           
          return data

        inputs = [pb_utils.Tensor("input_ids", cast(encoded_input['input_ids'], model_mode)),
                  pb_utils.Tensor("attention_mask", cast(encoded_input['attention_mask'], model_mode)), 
                  pb_utils.Tensor('token_type_ids', cast(encoded_input['token_type_ids'], model_mode))]
       
        # let's infer the model locally on server side
        inference_request = pb_utils.InferenceRequest(
          model_name=model_dict[model_mode],
          requested_output_names= [self.output_names[0],],
          inputs=inputs)

        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output0 = pb_utils.get_output_tensor_by_name(inference_response, self.output_names[0])

        logits = from_dlpack(output0.to_dlpack())


    #PostProcessing steps
        preds = tensor2list(torch.argmax(logits, dim = -1))
        processed_results = postprocessing(preds, {"0": "negative", "1": "positive"})

        inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor(
                self.s1_output_names[0], 
                np.array(processed_results[0], dtype=object)
                )])

        responses.append(inference_response)
      return responses

    def finalize(self):

      print("Stay cool. Byebye!")