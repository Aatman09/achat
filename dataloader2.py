import numpy as np 
import os
import torch 
import glob
from torch.utils.data import DataLoader


data_dir = "/home/aries/achat/fineweb_shards"

class ShardedDataset(DataLoader):

    def __init__(self, data_dir: str, B: int, T: int):
        self.B = B
        self.T = T
        
        self.shard_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        assert len(self.shard_files) > 0, f"No .npy files found in {data_dir}"
        
        self.shard_idx = 0
        
        # Load the very first shard
        self._load_shard(self.shard_idx)
        
        
        self.current_pos = len(self.tokens) - (B * T * 2)

    def _load_shard(self , shard_idx):
        filepath = self.shard_files[shard_idx]
        self.tokens = np.memmap(filepath , dtype = np.uint16 ,mode= 'r')

    def next_batch(self):

        B  , T = self.B , self.T
        if self.current_pos +B*T+1 > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shard_files)
            self._load_shard(self.shard_idx)
            self.current_pos = 0
        buffer = self.tokens[self.current_pos : self.current_pos + B*T+1]
        x = torch.tensor(buffer[:-1].astype(np.int64) , dtype = torch.long)
        y = torch.tensor(buffer[1:].astype(np.int64) , dtype = torch.long)


        x = x.view(B,T)
        y = y.view(B,T)

        self.current_pos += B*T

        return x ,y
    


