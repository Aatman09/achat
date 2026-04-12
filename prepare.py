import numpy as np 
import tqdm 
from datasets import load_dataset
from tqdm import tqdm 
import tiktoken 
import os

DATASET_NAME = "HuggingfaceFW/fineweb-edu"
SAMPLE_NAME = "sample-10BT"
SHARD_SIZE = 100_000_000 
OUTPUT_DIR = "fineweb_shards"


os.makedirs(OUTPUT_DIR , exist_ok = True)

enc = tiktoken.get_encoding('gpt2')

dataset = load_dataset(DATASET_NAME, name = SAMPLE_NAME , split='train' , streaming=True)

token_buffer = np.empty(SHARD_SIZE , dtype = np.uint16)
buffer_idx = 0 
shard_idx = 0

for doc in tqdm(dataset):
    tokens = enc.encode_ordinary(doc['text'])
    tokens.append(enc.eot_token)

    for token in tokens:
        token_buffer[buffer_idx] = token
        buffer_idx +=1

        if buffer_idx == SHARD_SIZE:
            filename = os.path.join(OUTPUT_DIR , f"fineweb_edu_{shard_idx:06d}.npy")
            np.save(filename, token_buffer)
            print(f'\nSaved {filename}')

            buffer_idx = 0
            shard_idx+=1
            token_buffer = np.empty(SHARD_SIZE , dtype=np.uint16)

        
            if shard_idx == 10 :
                print('DONE')
                exit(0)