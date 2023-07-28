from datasets import load_dataset

data = load_dataset("multi_type_note_builder.py", split="dummy")
for i in range(len(data)):
    print(data[i])