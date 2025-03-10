import torch
import h5py
from einops import rearrange, reduce
from rich.progress import track

import json
import argparse

from diffusion import Diffusion

from Unet import UnetND
from CUBEPlot import plot_cube, plot_grid
from ROGSHUtils import ROGSH432
GSH = ROGSH432()

from functools import partial

from MOSMKernel import MultiouputSpectralMixtureKernel, LHSParams

from GRFSampler import TwoPCorrelation

results=h5py.File("CSResults.h5","a")
nsamples = 6

#################### Utility Functions

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

def BM2SM(bmasks,structs):
    assert bmasks.shape == structs.shape

    masks = torch.full_like(structs,float('nan'))

    masks[bmasks == 1] = structs[bmasks==1]

    return masks


#################### Load Test Structures


TSFile = h5py.File("data/Test_Structures.h5","r")

TS1, _ = GSH.CHullProj(TSFile["CS1"][:])
TS2, _ = GSH.CHullProj(TSFile["CS2"][:])

TS1 = rearrange(torch.tensor(TS1,device="cuda",dtype=torch.float), "x y z c -> 1 c x y z")
TS2 = rearrange(torch.tensor(TS2,device="cuda",dtype=torch.float), "x y z c -> 1 c x y z")

TS1Stats = TwoPCorrelation(rearrange(TS1,"1 c x y z -> x y z c"))
TS2Stats = TwoPCorrelation(rearrange(TS2,"1 c x y z -> x y z c"))


#################### Set Up Diffusion Prior

config_dir = "experiments/DM_Multi_2025_01_27_14_24_18_R0/config.json"
model_dir = "experiments/DM_Multi_2025_01_27_14_24_18_R0/logs/version_0/checkpoints/epoch=3-step=656.ckpt"

#load config
with open(config_dir,'r') as f:
    config = json.load(f)
    config = dict2namespace(config)

if config.periodic:
    from UnetP import UnetND
else:
    from Unet import UnetND

unet = UnetND(
    config.udim,
    dim_mults= tuple(config.channels),
    channels=config.in_channels
)
    
unet.sample_size = TS1.shape[-1]
unet.in_channels = config.in_channels
unet.dims=config.dims

diffusion = Diffusion.load_from_checkpoint(model_dir,model=unet, bins_max = config.sample_steps)

#################### MASKED SAMPLING CaseStudy


################ MAKE MASKS

#Missing At Random
MAR = torch.randint(2,TS1.shape,dtype=torch.int)

#Every Other Slice
EOS = torch.zeros(TS1.shape,dtype=torch.int)
for i in range(TS1.shape[2]):
    if i%2 == 0:
        EOS[:,:,i,:,:] = 1

#Every Third Slice
ETS = torch.zeros(TS1.shape,dtype=torch.int)
for i in range(TS1.shape[2]):
    if i%3 == 0:
        ETS[:,:,i,:,:] = 1

#Every Fourth Slice
EFS = torch.zeros(TS1.shape,dtype=torch.int)
for i in range(TS1.shape[2]):
    if (i+1)%4 == 0:
        EFS[:,:,i,:,:] = 1

CHK = torch.ones(TS1.shape,dtype=torch.int)
CHK[:,:,:32,:32,:32] = 0
CHK[:,:,-32:,-32:,-32:] = 0

#Combo
CMB = torch.logical_and(torch.logical_and(CHK, EOS),MAR)*1

#Orthogonal Slices
OTH = torch.zeros(TS1.shape,dtype=torch.int)
OTH[:,:,:,:,-1] = 1
OTH[:,:,:,-1,:] = 1
OTH[:,:,0,:,:] = 1

################ Write Cond Function

def masked_cond_fn(images,masks):
    images[torch.logical_not(torch.isnan(masks))] = masks[torch.logical_not(torch.isnan(masks))]
    return images

################ Run Case Study


CS_List = [

        {
        "savedir":"figs/CaseStudies/CS1/TS1_EOS",
        "mask": EOS,
        "steps":100,
        "skip": 75
    },
    {
        "savedir":"figs/CaseStudies/CS1/TS1_ETS",
        "mask": ETS,
        "steps":100,
        "skip": 75
    },
        {
        "savedir":"figs/CaseStudies/CS1/TS1_CMB",
        "mask": CMB,
        "steps":100,
        "skip": 75
    },
    {
        "savedir":"figs/CaseStudies/CS1/TS1_OTH",
        "mask": OTH,
        "steps":100,
        "skip": None
    },
]

"""
    {
        "savedir":"CS1/TS1_MAR",
        "mask": MAR,
        "steps":50,
        "skip": None
    },
    {
        "savedir":"CS1/TS1_CHK",
        "mask": CHK,
        "steps":50,
        "skip": None
    },
    {
        "savedir":"CS1/TS1_EFS",
        "mask": EFS,
        "steps":100,
        "skip": 75
    },

"""


'''
for CS in CS_List:
    samples = []
    for i in track(range(nsamples)):
        BMASK = CS["mask"]

        #plot_cube(rearrange(BMASK[0,0].repeat(3,1,1,1),"c x y z -> x y z c"),CS["savedir"]+"_mask.png")

        masks = BM2SM(
                    BMASK,
                    TS1
                    )
        
        partial_masked_cond_fn = partial(masked_cond_fn, masks = masks)
        sample = diffusion.sample(steps=CS["steps"], cond_fn = partial_masked_cond_fn)

        if CS["skip"] is not None:
            sample = diffusion.sample(steps=CS["steps"], skip = CS["skip"], images=sample)

        sample = rearrange(sample[0], "c x y z -> x y z c")

        samples.append(sample)

    samples = torch.stack(samples,dim=0).detach().cpu().numpy()
    results.create_dataset(CS["savedir"],data=samples , compression="gzip")

    #sample, idx = GSH.CHullProj(sample)
    #sample_euler = GSH.euler[idx]

    #plot_cube(sample*.5 + .5,CS["savedir"]+"_rogsh.png")
    #pc_ipf(sample_euler,CS["savedir"]+"_ipf.png")
    #print(f"Saving to:"+CS["savedir"])
'''


#################### Orthogonal 2PS Reconstruction CaseStudy


################ Generate 2PS for testing


for target_stats, save_dir in zip([TS1Stats],["CS2/TS1-2"]):#zip([TS1Stats,TS2Stats],["CS2/TS1","CS2/TS2"]):
    samples = []
    for i in track(range(nsamples)):
        steps = 25

        '''
        plot_cube(
            torch.fft.fftshift(TS2Stats[...,0]).detach().cpu().numpy(),
            "figs/CaseStudies/CS2/TS2_STS.png",
            mode = "scalar",
            kind = "interior",
            cbar = True,
            elev=20,
            azim=-40
            )
        '''

        target = torch.stack([target_stats[0,:,:,:],target_stats[:,0,:,:],target_stats[:,:,0,:]])
        target_mean = reduce(TS2,"b c x y z -> c","mean")

        def ortho_loss(images):
            stats = TwoPCorrelation(rearrange(images,"1 c x y z -> x y z c"))
            stats = torch.stack([stats[0,:,:,:],stats[:,0,:,:],stats[:,:,0,:]])
            loss = torch.nn.functional.mse_loss(stats,target)
            
            return loss

        ortho_loss_grad = torch.func.grad_and_value(ortho_loss)

        def opt_step(images,alpha):
            with torch.no_grad():
                grad, loss = ortho_loss_grad(images)
            return images - alpha*grad, loss

        def ortho_stats_cond_fn(images):
            loss = 1e10

            thresh = 1e-8 #( (i-1) / (steps-1) )*(1e-6 - 5e-8) + 5e-8

            while loss > thresh:
                images, loss = opt_step(images, 7_000_000)
                loss = loss.item()

            mean_shift = target_mean-reduce(images,"b c x y z -> c", "mean")
            images = rearrange(images,"b c x y z -> b x y z c") + mean_shift
            images = rearrange(images, " b x y z c -> b c x y z")
            return images
            
        sample = diffusion.sample(steps=steps, cond_fn = ortho_stats_cond_fn)
        plot_cube(rearrange(sample[0], "c x y z -> x y z c").detach().cpu().numpy()*.5 + .5,"test.png")

        samples.append(sample[0])
        exit()

    samples = samples = torch.stack(samples,dim=0).detach().cpu().numpy()
    results.create_dataset(save_dir,data=samples , compression="gzip")


    #plot_cube(rearrange(sample[0], "c x y z -> x y z c").detach().cpu().numpy()*.5 + .5,"figs/CaseStudies/CS2/TS2_OPT.png")

    #GenStats = TwoPCorrelation(rearrange(sample,"1 c x y z -> x y z c"))

    '''
    plot_cube(
        torch.fft.fftshift(GenStats[...,0]).detach().cpu().numpy(),
        "figs/CaseStudies/CS2/TS2_GTS.png",
        mode = "scalar",
        kind = "interior",
        cbar = True,
        elev=20,
        azim=-40
        )
    '''




