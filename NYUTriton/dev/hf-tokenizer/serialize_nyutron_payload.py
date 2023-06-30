"""
This script will create a JSON payload for sending as a post-request to NYUtriton. You can specify
a string payload and a model and it will format the payload and provdie you with a simple curl request you can
just copy and paste into the browser to test. See here:

$ python serialize_txt.py -p "test payload is test payload" -m hf-inference
curl -X POST http://IP:8000/v2/models/hf-inference/versions/1/infer --data-binary "@postrequest.bin" --header "Inference-Header-Content-Length: 167"

$ curl -X POST http://IP:8000/v2/models/hf-inference/versions/1/infer --data-binary "@postrequest.bin" --header "Inference-Header-Content-Length: 167"
{"model_name":"hf-inference","model_version":"1","outputs":[{"name":"textout","datatype":"BYTES","shape":[],"data":["{'label': 'LABEL_0', 'score': 0.5632727146148682}"]}]}

"""

from re import A
import struct
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Praser for the payloads")

# Add the arguments
parser.add_argument('-p', "--payload",
    type=str,
    default="by the end of no such thing the audience , like beatrice , has a watchful affection for the monster.\n",
    help="Text payload for nyutriton.")

parser.add_argument('-m', "--model",
    type=str,
    default="nyutron-readmission",
    help="model name as defined in NYUtriton")


def build_payload(text, model_name):
    """Creates a JSON payload for NYUtriton

    Takes a string input and returns a UTF8 encoded, serialzed JSON for sending via curl to NYUtriton.
    This will save the JSON header separately from the total payload to allow calculating the header 
    length and payload length separately which is necessary for the curl request.

    Args:
        text: str, a string 

    Returns:
        A json file postrequest.bin that can be used with NYUtriton

    Errors:
        TypeError: Raised when the payload isn't a string.
    """    
    #Encode payload.
    # <I means little-endian unsigned integers, followed by the number of elements
    text_b: bytes = text.encode("UTF-8")
    print("Text length:{0}".format(len(text_b)))

    # define JSON structure... these names need to match the model. 
    json_struct='{"inputs":[{"name":"rawtext","shape":[1,1],"datatype":"BYTES","parameters":{"binary_data_size":'+str(len(text_b)+4)+'}}],"outputs":[{"name":"logits","parameters":{"binary_data":false}}]}'
    json_struct_b = json_struct.encode("UTF-8")
    print("Total JSON payload length:{0}".format(len(json_struct_b)))
    with open('header.bin', 'wb') as f:f.write(json_struct_b)

    #Build payload
    post_request = json_struct_b+struct.pack("<I", len(text_b))+text_b
    with open('postrequest.bin', 'wb') as f:f.write(post_request)

    #Print out post-request to terminal for user QOL:
    print("""
    curl -X POST http://IP:8000/v2/models/{0}/versions/1/infer --data-binary "@postrequest.bin" --header "Inference-Header-Content-Length: {1}"
    """.format(model_name,len(json_struct_b))) 

    return

if __name__ == '__main__':
    args = parser.parse_args()
    
    build_payload(
        text = args.payload,
        model_name = args.model)