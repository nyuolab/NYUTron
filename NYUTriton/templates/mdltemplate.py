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
    """
    model_base = the directory in the production area of triton /models in teh container and /tritonmodelrepo in the main directory where we store the accerelated version of the model there will be a second repo named "model_name_onnx" where the business logic version lives that integrates the tokenizer
    Note that we are pulling in the model architecture name from the HF config.json file
    during the deploy_model.py script and using that to set the input and output names.
    """
    def __init__(self):
        ## TOIMPROVE performance later
        # self.dl = create_infer_dataloader
        self.tokenizer = AutoTokenizer.from_pretrained("/models/{{model_name}}_{{model_type}}/1/hf")
        self.model_type = '{{model_type}}'
        self.model_base = '{{model_name}}_base'

        #These are the overall Triton pipeline input and output names
        self.s1_input_names = ['{{input_name}}',]
        self.s1_output_names = ['{{output_name}}',]


    def execute(self, requests, verbose=True):
      
      responses = []

      for request in requests:

          # get the text payload
          raw_strings = pb_utils.get_input_tensor_by_name(request, self.s1_input_names[0]).as_numpy()
          text = raw_strings.item().decode()

          # ToDo: this should automatically look at the task type from HF and define the input setup, ditto for inputs below
          # call the tokenizer on the server side
          # NOTA BENE --> PREPROCESSING STEPS OCCUR HERE
          encoded_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            truncation = True,
            padding = "max_length",
            return_attention_mask = True,
            max_length = {{max_seq_length}},
            return_tensors = "np") #this has to be np so returns numpy arrays instead of pytorch tensors triton likes numpy

          # conditional cast to INT32
          def cast(data):
            """
            cast conditionally to INT32 since TRT does not like INT64
            """
            if self.model_type == 'trt':
              return data.astype(np.int32)
                      
            return data

          #Kind of hacky, but a logical statement to let template know this is the right input architecture for the loaded HF model type
          #There needs to be several of these, one for each type of HF model and input architecture
          #I think the next one should be a generative input head compatible for GPT2 like models
          if '{{model_architecture}}' == 'BertForSequenceClassification':
            inputs = [pb_utils.Tensor("input_ids", cast(encoded_input['input_ids'])),
                      pb_utils.Tensor("attention_mask", cast(encoded_input['attention_mask'])),
                      pb_utils.Tensor('token_type_ids', cast(encoded_input['token_type_ids']))]
        
          # let's infer the model locally on server side
          inference_request = pb_utils.InferenceRequest(
            model_name=self.model_base,
            requested_output_names=['logits',], #has to be the HF defined ouptut name
            inputs=inputs)

          inference_response = inference_request.exec()

          if inference_response.has_error():
              raise pb_utils.TritonModelException(inference_response.error().message())
          else:
              # Extract the output tensors from the inference response.
              output0 = pb_utils.get_output_tensor_by_name(inference_response, self.s1_output_names[0])

          logits = from_dlpack(output0.to_dlpack())


        #PostProcessing steps
          preds = tensor2list(logits)
          #processed_results = postprocessing(preds, {"0": "negative", "1": "positive"})

            # inference_response = pb_utils.InferenceResponse(
            #         output_tensors=[pb_utils.Tensor(
            #         self.s1_output_names[0], 
            #         np.array(processed_results[0], dtype=object)
            #         )])
          inference_response = pb_utils.InferenceResponse(
                  output_tensors=[pb_utils.Tensor(
                  self.s1_output_names[0], 
                  np.array(preds)
                  )])

          responses.append(inference_response)
      return responses

    def finalize(self):
      print("Stay cool. Byebye!")