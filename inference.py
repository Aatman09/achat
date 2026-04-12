import torch
import tiktoken
from model import GPT  

# 1. Hardware Setup (RTX 4050 optimized)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Booting up neural network on: {device}...")

# 2. Load the Architecture and Weights
model = GPT.from_pretrained('gpt2')

# Load the raw dictionary from the SSD
raw_state_dict = torch.load("shakespeare_gpt.pth", map_location=device, weights_only=True)

# Create a clean dictionary by stripping the "_orig_mod." prefix
clean_state_dict = {}
for key, value in raw_state_dict.items():
    if key.startswith("_orig_mod."):
        clean_key = key[10:] 
        clean_state_dict[clean_key] = value
    else:
        clean_state_dict[key] = value

# Load the clean dictionary into your model
model.load_state_dict(clean_state_dict)
model.to(device)

# CRITICAL SYSTEM COMMANDS:
model.eval()  # Tells layers like LayerNorm to stop training and lock their stats
model = torch.compile(model) # Keep the Ada Lovelace speed boost!

# 3. Prepare the Tokenizer and Prompt
enc = tiktoken.get_encoding('gpt2')
prompt = """Question: What is the capital of France?
Answer: Paris

Question: What is the capital of Japan?
Answer: Tokyo

Question: What is the capital of the USA?
Answer:"""
print(f"\nPrompt: {prompt}\n")

# Convert string to tokens, then to a PyTorch tensor, and add a Batch dimension (1, T)
tokens = enc.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

# 4. The Autoregressive Generation Loop
max_new_tokens = 50 

# Disable gradient tracking. We aren't doing calculus anymore.
with torch.no_grad(): 
    # Use your RTX 4050's hardware math format
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        
        for step in range(max_new_tokens):
            # GPT-2 can only look back 1024 tokens. If our sequence gets too long, crop it.
            idx_cond = x[:, -1024:]
            
            # Forward pass: Ask the model to predict the next word
            logits, _ = model(idx_cond)
            
            # Pluck the last token's logits (Shape: [1, 50257])
            logits = logits[:, -1, :] 
            
            # --- THE SAMPLING BLOCK ---
            
            # Hyperparameters for creativity
            temperature = 0.8 # 1.0 = normal, >1.0 = chaotic, <1.0 = predictable
            top_k = 50        # Only consider the top 50 most likely words
            
            # Apply temperature to scale the raw logits
            logits = logits / temperature
            
            # Find the threshold value of the 50th most likely token
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            
            # Masking: Force everything outside the Top K to negative infinity 
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert the raw mathematical logits into percentages (0.0 to 1.0)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Roll the dice! Sample randomly based on those percentages
            next_token = torch.multinomial(probs, num_samples=1)
            
            # --------------------------
            
            # Glue the new token to the end of our running sequence
            x = torch.cat((x, next_token), dim=1)

# 5. Decode the brainwaves back to English
generated_tokens = x[0].tolist()
generated_text = enc.decode(generated_tokens)

print("--- GENERATED TEXT ---")
print(generated_text)
print("----------------------")