import tiktoken
import torch


class Dataloaderlite:
    def __init__(self  , B  : int  , T : int ,  filepath: str = "/home/aries/achat/input.txt" ):
        self.B = B
        self.T = T 
        with open(filepath , "r") as f:
            txt = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(txt)

        self.tokens = torch.tensor(tokens , dtype= torch.long)

        self.curr_pos = 0

    def next_batch(self):
        B , T = self.B , self.T

        buff = self.tokens[self.curr_pos : self.curr_pos + B*T +1]

        x = buff[:-1]
        y = buff[1:]

        x = x.view(B, T)
        y = y.view(B, T)
        self.curr_pos += B*T 


        if self.curr_pos + (B*T)+1 >= len(self.tokens):
            self.curr_pos = 0

        return x , y
    
