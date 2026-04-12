import torch 
from dataloader2 import ShardedDataset
from model import GPT
import time
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
filepath = "/home/aries/achat/fineweb_shards"


model = GPT.from_pretrained ('gpt2')
model.to(device)
model = torch.compile(model) # further reduced time from .5 sec to .28 sec but only works on linux due to trition kernel


torch.set_float32_matmul_precision('medium')
grad_acc_step  = 8
B = 8 
T = 512
epochs = 500
train_dataset = ShardedDataset(filepath , B , T)

optimizer = torch.optim.AdamW(model.parameters() , lr = 3e-4 , betas=(0.9,0.95))

x , y = train_dataset.next_batch()
x , y = x.to(device) , y.to(device)

loss_graph = []
optimizer.zero_grad() 
for step in range(epochs):
    
    t0 = time.time()

    x , y = train_dataset.next_batch()
    x , y = x.pin_memory().to(device , non_blocking=True) , y.pin_memory().to(device , non_blocking=True)

    # optimizer.zero_grad()
    with torch.autocast(device_type=device , dtype = torch.bfloat16): # made the training time from 15 to 17 sec per epoch to 3 - 5 sec
        logits , loss = model(x , targets= y)
    loss_graph.append(loss.item())
    loss = loss / grad_acc_step


    loss.backward()
    norm = -1.0
    if (step+1) % grad_acc_step == 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        
        t1 = time.time() - t0
        total_tokens = train_dataset.B * train_dataset.T * grad_acc_step
        tokens_per_second = total_tokens / t1
        
        print(f"Step : {step} | loss : {loss.item() * grad_acc_step:.4f} | time : {t1*1000:.2f}ms  | norm  : {norm:.4f} | toks/sec : {tokens_per_second:.2f} t/s")
        t0 = time.time()

loss_graph = np.array(loss_graph)

torch.save(model.state_dict(), "shakespeare_gpt.pth")
print(f"Model Saved with loss of : {loss}")

plt.plot(loss_graph)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
