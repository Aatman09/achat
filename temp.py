import json
import urllib.request
import tiktoken
import numpy as np
import torch

# 1. Download the famous Stanford Alpaca dataset (Tiny version for testing)
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
print("Downloading Alpaca dataset...")
urllib.request.urlretrieve(url, "alpaca.json")

with open("alpaca.json", "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} instruction examples.")

# 2. Look at a single example to see the structure
sample = dataset[0]
print("\n--- RAW JSON DATA ---")
print(f"Instruction: {sample['instruction']}")
print(f"Input: {sample['input']}") # Sometimes empty
print(f"Output: {sample['output']}")
print("---------------------\n")

# 3. The Formatting Template
# This is the strict string format we will force the model to learn
def format_prompt(example):
    if example['input'] == "":
        return f"User: {example['instruction']}\nAssistant: {example['output']}<|endoftext|>"
    else:
        return f"User: {example['instruction']}\nContext: {example['input']}\nAssistant: {example['output']}<|endoftext|>"

formatted_sample = format_prompt(dataset[0])
print("--- EXACT STRING THE GPU WILL SEE ---")
print(formatted_sample)