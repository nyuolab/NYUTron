import numpy as np
import torch
from nemo.utils import logging
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.collections import nlp as nemo_nlp

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

# import onnxruntime

## Patching DatasetClass
from nemo.collections.nlp.data.text_classification import TextClassificationDataset

class TextClassificationDatasetPatched(TextClassificationDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def _collate_fn(self, batch):
        """collate batch of input_ids, segment_ids, input_mask, and label
        Args:
            batch:  A list of tuples of (input_ids, segment_ids, input_mask, label).
        """
        max_length = self.max_seq_length
        for input_ids, segment_ids, input_mask, label in batch:
            if len(input_ids) > max_length:
                max_length = len(input_ids)

        padded_input_ids = []
        padded_segment_ids = []
        padded_input_mask = []
        labels = []
        for input_ids, segment_ids, input_mask, label in batch:
            if len(input_ids) < max_length:
                pad_width = max_length - len(input_ids)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_width], constant_values=self.pad_id))
                padded_segment_ids.append(np.pad(segment_ids, pad_width=[0, pad_width], constant_values=self.pad_id))
                padded_input_mask.append(np.pad(input_mask, pad_width=[0, pad_width], constant_values=self.pad_id))
            else:
                padded_input_ids.append(input_ids)
                padded_segment_ids.append(segment_ids)
                padded_input_mask.append(input_mask)
            labels.append(label)

        return (
            torch.LongTensor(padded_input_ids),
            torch.LongTensor(padded_segment_ids),
            torch.LongTensor(padded_input_mask),
            torch.LongTensor(labels),
        )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def postprocessing(results, labels):
    return [labels[str(r)] for r in results]

def create_infer_dataloader(tokenizer, queries):
    batch_size = len(queries)
    # batch_size = 1
    dataset = TextClassificationDatasetPatched(tokenizer=tokenizer, queries=queries, max_seq_length=192)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

def triton_inferer(input_data, model_name, input_names, output_names, headers=None):
    inputs = []
    
    inputs.append(httpclient.InferInput(input_names[0], [4, 192], "INT32"))
    inputs.append(httpclient.InferInput(input_names[1], [4, 192], "INT32"))
    inputs.append(httpclient.InferInput(input_names[2], [4, 192], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input_data[0].cpu().numpy().astype('int32'), binary_data=True)
    inputs[1].set_data_from_numpy(input_data[1].cpu().numpy().astype('int32'), binary_data=True)
    inputs[2].set_data_from_numpy(input_data[2].cpu().numpy().astype('int32'), binary_data=True)

    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=None,
                                  query_params=query_params,
                                  headers=headers)
    print(results)
    return results.as_numpy(output_names[0])

if __name__ == '__main__':

    queries = ["by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .",
    "director rob marshall went out gunning to make a great one .",
    "uneasy mishmash of styles and genres .",
    "I love exotic science fiction / fantasy movies but this one was very unpleasant to watch . Suggestions and images of child abuse , mutilated bodies (live or dead) , other gruesome scenes , plot holes , boring acting made this a regretable experience , The basic idea of entering another person's mind is not even new to the movies or TV (An Outer Limits episode was better at exploring this idea) . i gave it 4 / 10 since some special effects were nice ."]

    # Get Model MetaData

    triton_client = httpclient.InferenceServerClient(url='localhost:8000') # select IP for TRITON
    model_name = 'model_trt' # select model on the TRITON server
    metadata = triton_client.get_model_metadata(model_name,
                                                    query_params={
                                                        'test_1': 1,
                                                        'test_2': 2
                                                    })
    if not (metadata['name'] == model_name):
        print("FAILED : get_model_metadata")
    print(metadata)
    # PreProcessing (tokenize, create batch etc)
    
    infer_dataloader = create_infer_dataloader(nemo_nlp.modules.get_tokenizer(tokenizer_name="bert-base-uncased"), queries)
    # ort_session = onnxruntime.InferenceSession("trained-model.onnx")
    logging.info('The prediction results of some sample queries with the trained model:')

    # Inference with TRITON

    input_names = ['input_ids','attention_mask','token_type_ids']
    output_names = ['logits']

    for batch in infer_dataloader:
        ologits = triton_inferer(batch, model_name, input_names, output_names) # Inference Step

        print(ologits)
        #PostProcessing steps
        logits = torch.from_numpy(ologits)
        preds = tensor2list(torch.argmax(logits, dim = -1))
        processed_results = postprocessing(preds, {"0": "negative", "1": "positive"})

        logging.info('The prediction results of some sample queries with the trained model:')
        for query, result in zip(queries, processed_results):
            # logging.info(f'Query : {query}')
            logging.info(f'Predicted label: {result}')
        