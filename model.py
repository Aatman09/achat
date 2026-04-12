import torch 
import torch.nn as nn 
import torch.nn.functional as F
from dataclasses import dataclass



@dataclass
class GPTConfig:
    block_size : int = 1024 
    vocab_size : int = 5027
    n_layer : int = 12
    n_embd : int = 768
    n_head : int = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0  
        """we are making the Wq , Wk , Wv matrix in this we are making a general lineat layer then 
           we will split that into parts of q, k , v  , c_proj will be used after  combining the 
           embedding from the mulihead attention"""
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd , config.n_embd)

        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size , config.block_size).view(1 , 1 , config.block_size , config.block_size))
        )

    def forward(self , x):
        B,T, C = x.shape
        qkv = self.c_attn(x)
        q , k  , v = qkv.split(self.n_embd , dim = 2)

        q = q.view(B, T ,  self.n_head , self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head , self.head_size).transpose(1,2)
        v = v.view(B ,T , self.n_head , self.head_size).transpose(1,2)
        
        out = F.scaled_dot_product_attention(q,k,v,is_causal=True) # reduced training time from 5 sec to .5 sec
        # scale = self.head_size ** -0.5
        # attn = (q @ k.transpose(-2 , -1)) *scale
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # out = attn @ v


        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out)
        return out
    
class MLP(nn.Module):
    def __init__(self , config : GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd , config.n_embd)
    
    def forward(self , x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 


class Block(nn.Module):

    def __init__(self , config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self , x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
    
class GPT(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),

        ))

        self.lm_head = nn.Linear(config.n_embd , config.vocab_size, bias=False)


    def forward(self ,idx , targets=None ):

        B,T = idx.size()
        assert T <= self.config.block_size ,f"Cannot forward sequence of length {T} , block_size is {B}"

        pos = torch.arange(0 , T , dtype=torch.long , device = idx.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(idx)
        x = tok_embd + pos_embd

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x)

        if targets is not None:
            B,T,C = logits.shape
            logits_view = logits.view(B*T , C)
            tragets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view , tragets_view)


         
        return logits , loss 


    @classmethod
    def from_pretrained(cls , model_type):
        
        assert model_type in {"gpt2" , "gpt2-medium" , "gpt2-large" , "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2" :   dict(n_embd =  768, n_head = 12, n_layer=12),
            "gpt2-medium" :   dict(n_embd =  1024, n_head = 16, n_layer=24),
            "gpt2-large" :   dict(n_embd =  1280, n_head = 20, n_layer=36),
            "gpt2-xl" :   dict(n_embd =  1600, n_head = 25, n_layer=48)
        }[model_type]


        config_args["vocab_size"] = 50257 # changed to be divisible by 4 , 8 , 16 , 32x
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict() 
        # print(sd)
        sd_keys = sd.keys()
        # print(sd_keys)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].t().shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ----------------------------------------------------------------------
