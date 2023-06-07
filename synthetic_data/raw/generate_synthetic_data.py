import openai
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
examples = []

# prompt from https://platform.openai.com/examples/default-qa
for note_id in tqdm(np.arange(100)):  # 10
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt='I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: Give an example of clinical note. \nA:',
        temperature=0.01 * note_id,  # 0.1 * note_id,
        max_tokens=100,  # 200
        top_p=1,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"],
    )
    text = response["choices"][0]["text"]
    text = ":".join(text.split(":")[1:])
    if len(text) > 0:  # only include nonempty example
        examples.append(text)

df = pd.DataFrame({"text": examples})
print(df)
df.to_csv("synthetic_readmission_larger.csv")  # "synthetic_data.csv")
