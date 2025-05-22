'''
Code for implementing periodic padding using the 
PolyMicros foundation model.

An adaptation of Michael's code.
'''
import torch, numpy as np
import h5py
from einops import rearrange, reduce
from rich.progress import track

import json
import argparse

from diffusion import Diffusion
import matplotlib.pyplot as plt

from Unet import UnetND as UnetND_Nonperiodic
from UnetP import UnetND as UnetND_periodic

from CUBEPlot import plot_cube, plot_grid
from ROGSHUtils import ROGSH432
GSH = ROGSH432()

from functools import partial

from MOSMKernel import MultiouputSpectralMixtureKernel, LHSParams
from GRFSampler import TwoPCorrelation
import typing

results=h5py.File("CSResults.h5","a")
nsamples = 6

class ModularConditioner(object):
    '''
    wrapper to facilitate modular conditioning of PolyMicros.
    '''
    def __init__(
            self, 
            model: typing.Callable | str,
            symmetry_type: str='cubic',
        ):

        # symmetry information
        if symmetry_type.lower() == 'cubic':
            self.GSH = ROGSH432()
        else:
            raise AttributeError('Only cubic symmetry is supported right now.')

        # 

#################### Utility Functions

def load_polymicros(
        model_ckpt: str, 
        config: str, 
        default_spatial_voxelization: int=128,
        periodic: typing.Optional[bool]=None,
    ) -> Diffusion:
    '''
    load the PolyMicros diffusion model.

    :param periodic: load the model in periodic mode. If set to none, it will
                     utilize the option in the provided config file by defult.
    '''
    #load config
    with open(config,'r') as f:
        config = json.load(f)
        config = dict2namespace(config)

    periodic = periodic if periodic is not None else config.periodic
    UnetND = UnetND_Nonperiodic if not periodic else UnetND_periodic

    # initialize model
    unet = UnetND(
        config.udim,
        dim_mults= tuple(config.channels),
        channels=config.in_channels
    )
        
    unet.sample_size = default_spatial_voxelization # this is a very weird way of funneling info
    unet.in_channels = config.in_channels
    unet.dims=config.dims

    # loading the diffusion loader
    diffusion = Diffusion.load_from_checkpoint(
        model_ckpt,
        model=unet, 
        bins_max = config.sample_steps
    )
    return diffusion

def masked_cond_fn(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    '''
    mask type conditioning function. Honestly, not super sure what this is doing...
    '''
    images[torch.logical_not(torch.isnan(masks))] = masks[torch.logical_not(torch.isnan(masks))]
    return images

def mask_generating_helper_periodic_padding(
        base_shape: tuple[int],
        padding_width: tuple[int] | int,
        device: typing.Optional[torch.device] = None,
    ) -> torch.Tensor:
    '''
    helper function that creates a mask for using PolyMicros
    to periodically extend a microstructure.

    :param base_shape: the base shape of the structure you are
                       extending: [Batch, Channels, SPACEX, SPACEY, SPACEZ]
    '''
    assert len(base_shape) == 5, 'Only supports 3D structures in pytorch convention'
    padding_width = [padding_width for _ in range(3)] \
        if type(padding_width) is not tuple and type(padding_width) is not list \
        else padding_width 
    
    padded_shape = tuple(
        list(base_shape[:2]) + \
        [dim + 2*pad_width for dim, pad_width in zip(base_shape[2:], padding_width)]
    )
    center_slicers = [slice(pad_width, -pad_width) for pad_width in padding_width]

    # create a padded volume
    mask = torch.zeros(padded_shape, dtype=torch.int, device=device)

    # mask out center -- known values
    mask[..., *center_slicers] = 1.0

    # embedding function
    def embed_image(image):
        '''
        embeds the image
        '''
        embedded_image = torch.zeros_like(mask).float()
        embedded_image[..., *center_slicers] = image.clone()
        return embedded_image

    return mask.float(), embed_image


def random_other_mask_helpers():

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

def BM2SM(bmasks, structs):
    '''
    combines the structure you are targetting with the
    mask that you generated.
    '''
    assert bmasks.shape == structs.shape

    masks = torch.full_like(structs,float('nan'))

    masks[bmasks == 1] = structs[bmasks==1]

    return masks



def scrap_code():
    #################### Load Test Structures

    TSFile = h5py.File("data/Test_Structures.h5","r")

    TS1, _ = GSH.CHullProj(TSFile["CS1"][:])
    TS2, _ = GSH.CHullProj(TSFile["CS2"][:])

    TS1 = rearrange(torch.tensor(TS1,device="cuda",dtype=torch.float), "x y z c -> 1 c x y z")
    TS2 = rearrange(torch.tensor(TS2,device="cuda",dtype=torch.float), "x y z c -> 1 c x y z")

    TS1Stats = TwoPCorrelation(rearrange(TS1,"1 c x y z -> x y z c"))
    TS2Stats = TwoPCorrelation(rearrange(TS2,"1 c x y z -> x y z c"))


def visualize_data():
    sample = np.moveaxis(np.load('./experiments/cutoutcube.npy'), -1, 0)
    sample = torch.from_numpy(sample[:, 50:50+64, 120:120 + 64, 80:80 + 64, ...])
    print(sample.shape)

    f, axes = plt.subplots(1, 3, figsize=[15, 5])

    axes[0].imshow(torch.movedim(sample[:, 50, :, :], 0, -1))
    axes[1].imshow(torch.movedim(sample[:, :, 50, :], 0, -1))
    axes[2].imshow(torch.movedim(sample[:, :, :, 50], 0, -1))

    f.tight_layout()
    f.savefig('./figs/temp.png', dpi=300)

@torch.no_grad()
def main():
    #################### Set Up Diffusion Prior
    # Parameters
    config_dir = "./model_checkpoints/config.json"
    model_dir = "./model_checkpoints/PolyMicros.ckpt"
    savefolder = "figs/CaseStudies/CS1/TS1_EOS"
            #"mask": EOS,
    steps = 300
    skip = 75
    def_size = 64
    padding_width = 8
    periodic = True

    # loading the diffusion model
    polymicros = load_polymicros(
        model_ckpt = model_dir,
        config = config_dir,
        periodic=periodic,
        default_spatial_voxelization=def_size + padding_width * 2,
    )

    # load base sample and convert to ROGSH
    sample = np.load('./experiments/cutoutcube.npy')
    sample = sample[50:50+def_size, 120:120 + def_size, 80:80 + def_size, ...]
    print(sample.shape)
    sample = GSH.ROGSH(sample)
    sample = np.moveaxis(sample, -1, 0)[None, ...]
    sample = torch.from_numpy(sample).to(polymicros.device)

    # creating the padding mask
    padding_mask, embedder = mask_generating_helper_periodic_padding(
        base_shape = sample.shape,
        padding_width=padding_width,
        device = polymicros.device,
    )
    embedded_sample = embedder(sample)
    padding_mask = BM2SM(padding_mask, embedded_sample)

    partial_masked_cond_fn = partial(
        masked_cond_fn, 
        masks = padding_mask,
    )

    # conditional sampling
    sample = polymicros.sample(
        steps=steps,
        cond_fn=partial_masked_cond_fn,
    )

    # final unconditional refinement
    sample = polymicros.sample(
        steps=steps,
        skip = skip,
        images=sample
    )

    print(sample.shape)
    sample = np.moveaxis(sample[0].cpu().numpy(), 0, -1)
    print(sample.shape)

    f, axes = plt.subplots(1, 3, figsize=[15, 5])

    axes[0].imshow(sample[16, :, :])
    axes[1].imshow(sample[:, 16, :])
    axes[2].imshow(sample[:, :, 16])

    f.tight_layout()
    f.savefig('./figs/temp.png', dpi=300)



    exit()

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



@torch.no_grad()
def resolution_observation():
    # load base sample and convert to ROGSH
    sample = np.load('./experiments/cutoutcube.npy')
    from torchvision import transforms

    # Parameters
    # 226, 328, 264, 3
    top_resolution = 80
    final_resolutions = [128, 64, 32]

    X, Y, Z, C = sample.shape

    slice2d = sample[
        X//2, 
        Y // 2 - top_resolution // 2: Y // 2 + top_resolution // 2,
        Z // 2 - top_resolution // 2: Z // 2 + top_resolution // 2,
        :,
    ]
    slice2d = np.moveaxis(slice2d, -1, 0)[None, ...]

    print(slice2d.shape)

    slice_collection = [slice2d, ] + [transforms.Resize(rez, interpolation=transforms.InterpolationMode.NEAREST)(torch.from_numpy(slice2d)).numpy() for rez in final_resolutions]
    slice_collection = [np.moveaxis(sli[0], 0, -1) for sli in slice_collection]
    labels = ['Original', ] + [f'Rez: {rez}' for rez in final_resolutions]

    f, axes = plt.subplots(1, len(labels), figsize=[5 * len(labels), 5])
    for sli, label, ax in zip(slice_collection, labels, axes):
        ax.imshow(sli)
        ax.set_title(label)

    f.tight_layout()
    f.savefig(f'./figs/TG/resampling_{top_resolution}.png', dpi=300)

if __name__ == '__main__':
    #main()
    resolution_observation()
    #visualize_data()