"""KV-cached inference for the trained shakespeare GPT.

Loads the fine-tuned checkpoint (shakespeare_gpt.pth) on top of HF GPT-2 init,
then generates with KV caching — no recomputing K/V for past tokens each step.
"""
import torch
import tiktoken
from model_kv import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Booting up neural network on: {device}...")

model = GPT.from_pretrained('gpt2')

raw_state_dict = torch.load("shakespeare_gpt.pth", map_location=device, weights_only=True)
clean_state_dict = {}
for key, value in raw_state_dict.items():
    if key.startswith("_orig_mod."):
        clean_state_dict[key[10:]] = value
    else:
        clean_state_dict[key] = value

model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()

enc = tiktoken.get_encoding('gpt2')
prompt = """Question: What is the capital of France?
Answer: Paris

Question: What is the capital of Japan?
Answer: Tokyo

Question: What is the capital of the USA?
Answer:"""
print(f"\nPrompt: {prompt}\n")

tokens = enc.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=50)

generated_text = enc.decode(out[0].tolist())
print("--- GENERATED TEXT ---")
print(generated_text)
print("----------------------")
