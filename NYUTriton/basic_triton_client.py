import numpy as np
import argparse
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def postprocessing(results, labels):
    return [labels[str(r)] for r in results]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-f',
                        '--filename',
                        type=str,
                        required=False,
                        default='queries.txt',
                        help='Text file containing inputs for model')

    args = parser.parse_args()

    # Get Model MetaData
    triton_client = httpclient.InferenceServerClient(url=args.url) # select IP for TRITON
    
    # Set up input/output MetaData
    model_name = 'nemo-tokenizer'
    model_mode = 'model_trt'

    input_names = ['rawtext',]
    output_names = ['textout',]

    model_dict = {'model_onnx': 0,        # ONNX as is
                'model_trt': 1,    # TensorRT plan
                }

    with open(args.filename, 'r') as f:
        for input_data in f:
    
            input0 = np.array([[input_data] for i in range(1)], dtype=object)
            inputs = []
            
            inputs.append(httpclient.InferInput(input_names[0], input0.shape, "BYTES"))

            ## TODO defaulted everything to TRT Model for now
            # inputs.append(httpclient.InferInput('model_mode', [1, 1], "INT64"))

            outputs = [httpclient.InferRequestedOutput('textout'),]
            
            # Initialize the data
            inputs[0].set_data_from_numpy(input0)
            # inputs[1].set_data_from_numpy(np.array([[model_dict[model_mode]]], dtype=np.int64))

            results = triton_client.infer(model_name,
                                        inputs,
                                        outputs=outputs)
            out_strings = results.as_numpy(output_names[0])
            output = out_strings.item().decode()
            
            print(f'Query: {input_data}')
            print(f'Predicted label: {output}')
            print('------------------------------')
