import h5py
import torch
from torch.utils.data import Dataset
from einops import rearrange

class HDF5PartialDataset(Dataset):
    def __init__(self, path, prefix, channel_last = True):
        self.file = h5py.File(path,"r")
        self.group = self.file[prefix]
        self.channel_last = channel_last

    def __len__(self):
        return len(self.group)
    
    def __getitem__(self, index):

        struct = torch.tensor(self.group[str(index)][:])
        struct = rearrange(struct, "x y z c -> 1 c x y z")
        #struct = torch.nn.functional.interpolate(struct,scale_factor=(.5,.5,.5), mode='nearest')[0]
        return struct[0,:,:32,:32,:32]
        