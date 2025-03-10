import torch
import numpy as np
import h5py
from einops import rearrange
from rich.progress import track

from diffusion import Diffusion
import json
import argparse

from UnetP import UnetND
from CUBEPlot import plot_cube
from GRFSampler import StatisticsGenerator
from MOSMKernel import MultiouputSpectralMixtureKernel, LHSParams
from ROGSHUtils import ROGSH432
GSH = ROGSH432()


############################################Step 1: Setup parameters

dev = "cuda"
n_samples, n_mix, n_tasks, n_dim, edge = 2000, 4, 3, 3, 128

bounds_dict = {"cov":[1.5,5],
              "u":[-5,5],
              "w":[-.02,.02],
              "d":[-0.5,0.5],
              "p":[0,2*torch.pi],
              }

params_dict = LHSParams(n_samples, n_tasks, n_mix, n_dim, bounds_dict, dev = dev)

#set up mesh of positions
ratio = (1,1,1)
size = (edge*ratio[0],edge*ratio[1],edge*ratio[2])
xi = torch.linspace(-1*torch.pi*ratio[0],torch.pi*ratio[0],size[0])
yi = torch.linspace(-1*torch.pi*ratio[1],torch.pi*ratio[1],size[1])
zi = torch.linspace(-1*torch.pi*ratio[2],torch.pi*ratio[2],size[2])
pos = torch.stack(torch.meshgrid(xi,yi,zi,indexing="ij"),axis=-1)
pos = torch.fft.fftshift(pos,dim=tuple(range(0,n_dim)))
pos = pos.reshape(-1,n_dim).to(dev)
points = pos.shape[0]

############################################Step 2: Construct Stats, Sample, Save
'''

dataset = h5py.File("PM3D.h5","w")

z = 0
for s in track(range(n_samples)):
    stats = torch.stack([MultiouputSpectralMixtureKernel(pos,0,j,params_dict[s]).reshape(size)  for j in range(n_tasks)],-1)
    grf = StatisticsGenerator(stats, zmean = True)

    if True: #grf.min().item() > -1 and grf.max().item() < 1:
        dataset.create_dataset(f"{z}/params/cov",data=params_dict[s]["cov"].detach().cpu().numpy(), compression="gzip")
        dataset.create_dataset(f"{z}/params/u"  ,data=params_dict[s]["u"].detach().cpu().numpy()  , compression="gzip")
        dataset.create_dataset(f"{z}/params/w"  ,data=params_dict[s]["w"].detach().cpu().numpy()  , compression="gzip")
        dataset.create_dataset(f"{z}/params/d"  ,data=params_dict[s]["d"].detach().cpu().numpy()  , compression="gzip")
        dataset.create_dataset(f"{z}/params/p"  ,data=params_dict[s]["p"].detach().cpu().numpy()  , compression="gzip")

        dataset.create_dataset(f"{z}/stats/target" ,data=stats.detach().cpu().numpy() , compression="gzip")
        dataset.create_dataset(f"{z}/structures/grf0" ,data=grf.detach().cpu().numpy() , compression="gzip")
        shape = grf.shape
        z+=1

print(f"[INFO] Sucsessfully Generated {z} Structures")

'''

############################################Step 3: Locally Refine

'''
Run 1: Inconel635AM skip=10,modid=149 (Done)
Run 2: Inconel635AM skip=8,modid=149  (Done)
Run 3: Inconel635AM skip=0,modid=149  (Done)
Run 4: Inconel635AM skip=0,modid=157  (Done)
Run 4: Inconel635AM skip=0,modid=155  (Done)
Run 4: Inconel635AM skip=0,modid=141  (Done)

Run 5: AL2219R skip=0,modid=181 (Done)
Run 6: AL2219R skip=8,modid=181 (Done)
Run 6: AL2219R skip=0,modid=159 (Done)
Run 6: AL2219R skip=8,modid=159 (Done)
Run 6: AL2219R skip=0,modid=145 (Done)
Run 6: AL2219R skip=0,modid=179 (Done)

Run 6: Inconel718W skip=0,modid=199 (Done)
Run 6: Inconel718W skip=8,modid=199 (Done)
Run 6: Inconel718W skip=0,modid=167 (Done)
Run 6: Inconel718W skip=8,modid=167 (Done)
Run 6: Inconel718W skip=0,modid=195 (Done)
Run 6: Inconel718W skip=0,modid=197 (Done)

Run 7: Ti64R skip=8,modid=145 (Done)
Run 7: Ti64R skip=8,modid=157 (Done)
Run 7: Ti64R skip=0,modid=157 (Done)
Run 7: Ti64R skip=0,modid=145 (Done)
Run 7: Ti64R skip=0,modid=151 (Done)
Run 7: Ti64R skip=0,modid=139 (Done)

Run 8: NRL skip=8,modid=199 (Done)
Run 8: NRL skip=8,modid=189 (Done)
Run 8: NRL skip=0,modid=199 (Done)
Run 8: NRL skip=0,modid=189 (Done)
Run 8: NRL skip=0,modid=191 (Done)
Run 8: NRL skip=0,modid=187 (Done)
'''

###############Settings
config_dir = "config.json"
model_dir = "experiments/DM_Multi_2025_01_27_14_24_18_R0/logs/version_0/checkpoints/epoch=3-step=656.ckpt"
prefix = "Multi"
steps = 25
skip = 15
dev = "cuda"
modid= 0

###############

dataset = h5py.File("scratch/PM3D.h5","a")
keys = list(dataset.keys())

def dict2namespace(config: dict):
    """
    Converts a dictionary to a namespace
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

#load config
with open(config_dir,'r') as f:
    config = json.load(f)
    config = dict2namespace(config)

unet = UnetND(
    config.sample_size,
    dim_mults= tuple(config.channels),
    channels=config.in_channels
)
    
unet.sample_size = edge
unet.in_channels = config.in_channels
unet.dims=config.dims

diffusion = Diffusion.load_from_checkpoint(model_dir,model=unet, bins_max = config.sample_steps)

if skip == 0:
    keys = list(range(0,2200))

for key in track(keys[1:]):

    if skip == 0:
        sample = diffusion.sample(steps=steps)
    else:
        inital = dataset[key+f"/structures/grf0"][:]
        #plot_cube(inital.clip(-1,1)*.5 + .5,f"figs/LocalRef/{prefix}_{key}_GRF_Inital.png")
        inital = torch.tensor(inital,dtype=torch.float).to(dev)
        inital = rearrange(inital, "x y z c -> 1 c x y z")
        sample = diffusion.sample(steps=steps, images = inital, skip = skip)
    sample = rearrange(sample[0], "c x y z -> x y z c").detach().cpu().numpy()

    #plot_cube(sample*.5 + .5, f"figs/gen/{prefix}_{key}.png")
    exit()
    _, idx = GSH.CHullProj(sample)

    #save
    #dataset.create_dataset(str(key)+f"/structures/{prefix}_{skip}_{modid}_1" ,data=idx , compression="gzip", dtype=np.uint16)
    
    '''
    #reload code
    idx = dataset[key+f"/structures/{prefix}_{skip}_modid"]
    rogsh = GSH.rogsh[idx]
    plot_cube(rogsh*.5 + .5, f"figs/gen/{prefix}_{key}_reload.png")
    '''


