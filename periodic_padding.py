'''
Code for implementing periodic padding using the 
PolyMicros foundation model.

An adaptation of Michael's code.
'''
import orix.sampling
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
    # user warning
    if periodic:
        UserWarning('Usage of PolyMICROs with Periodic Boundary Conditions often requires more diffusion steps for stability.')

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

# just doing some learning
def cat(func):
    def cat_meow(*args, **kwargs):
        val = func(*args, **kwargs)
        return f'MEOW: {val}'
    return cat_meow

@cat
def summing(a, b):
    return a + b


def nanmask_generating_helper_periodic_padding(
        original_structure: torch.Tensor,
        padding_width: tuple[int] | int,
        overlap_width: tuple[int] | int,
    ) -> list[torch.Tensor, typing.Callable]:
    '''
    This helper function creates the nanmask for the 
    periodic padding.

    it also creates a method which will slice out the final microstructure
    from the generated synthetic structure.

    Critically, the periodic padding is performed using reshaping operations.

    padding is structured as follows:
     o-------o
     |  x----x
     |  |    |
     o  x----x

    :param base_shape: the base shape of the structure you are
                       extending: [Batch, Channels, SPACEX, SPACEY, SPACEZ]
    :param padding_width: ashjdlkjasdf
    :param overlap_width: ashjdlkjasdf
    '''
    base_shape = original_structure.shape
    assert len(base_shape) == 5, 'Only supports 3D structures in pytorch convention'
    padding_width = [padding_width for _ in range(3)] \
        if type(padding_width) is not tuple and type(padding_width) is not list \
        else padding_width 

    overlap_width = [overlap_width for _ in range(3)] \
        if type(overlap_width) is not tuple and type(overlap_width) is not list \
        else overlap_width 
    
    padded_shape = tuple(
        list(base_shape[:2]) + \
        [dim + over_width + pad_width for dim, pad_width, over_width in zip(base_shape[2:], padding_width, overlap_width)]
    )

    # helper slicers
    data_slicers = [
        slice(over_width + pad_width, over_width + pad_width + struct) \
            for pad_width, over_width, struct in zip(padding_width, overlap_width, base_shape[2:]) \
    ]
    periodic_slicers = [
        slice(over_width, over_width + pad_width + struct) \
            for pad_width, over_width, struct in zip(padding_width, overlap_width, base_shape[2:]) \
    ]

    def extract_final_microstructure(micro: torch.Tensor) -> torch.Tensor:
        '''
        extracts the final microstructure from the diffusion process.
        '''
        return micro[..., *periodic_slicers]

    # create the nan mask -- equivalent to the operations in def BM2SM
    masks = torch.full(padded_shape, float('nan'), device=original_structure.device)

    # placing center data
    masks[..., *data_slicers] = original_structure.clone()
    
    # padding edges
    # go in x then y then z
    masks[..., :overlap_width[0], :, :] = masks[..., -overlap_width[0]:, :, :]
    masks[..., :, :overlap_width[1], :] = masks[..., :, -overlap_width[1]:, :]
    masks[..., :, :, :overlap_width[2]] = masks[..., :, :, -overlap_width[2]:]

    # return the conditioning information as well
    def internal_cond_fn(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        conditioning
        '''
        # standard conditioning
        images[torch.logical_not(torch.isnan(masks))] = masks[torch.logical_not(torch.isnan(masks))]

        # continuity
        images[..., :overlap_width[0], :, :] = images[..., -overlap_width[0]:, :, :]
        images[..., :, :overlap_width[1], :] = images[..., :, -overlap_width[1]:, :]
        images[..., :, :, :overlap_width[2]] = images[..., :, :, -overlap_width[2]:]

        return images
    
    # return the conditioning information as well
    def second_stage_cond_fn(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        conditioning
        '''
        # continuity
        images[..., :overlap_width[0], :, :] = images[..., -overlap_width[0]:, :, :]
        images[..., :, :overlap_width[1], :] = images[..., :, -overlap_width[1]:, :]
        images[..., :, :, :overlap_width[2]] = images[..., :, :, -overlap_width[2]:]

        return images

    return masks, internal_cond_fn, second_stage_cond_fn, extract_final_microstructure


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
    steps = 100
    skip = int(0.85 * steps)
    def_size = 64
    padding_width = 18
    overlap_width = 10
    periodic = False
    old_mode = False

    # loading the diffusion model
    polymicros = load_polymicros(
        model_ckpt = model_dir,
        config = config_dir,
        periodic=periodic,
        default_spatial_voxelization=def_size + 2 * padding_width if old_mode else def_size + padding_width + overlap_width,
    )

    # load base sample and convert to ROGSH
    sample = np.load('./experiments/cutoutcube.npy')
    sample = sample[50:50+def_size, 120:120 + def_size, 80:80 + def_size, ...]
    sample = GSH.ROGSH(sample)
    sample = np.moveaxis(sample, -1, 0)[None, ...]
    sample = torch.from_numpy(sample).to(polymicros.device)
    sample_mean = sample.mean(dim=(-3, -2, -1), keepdims=True)
    sample = sample - sample_mean

    if not old_mode:
        padding_mask, masked_cond_fn, secondstage_cond_fn, final_micro_extractor = nanmask_generating_helper_periodic_padding(
            original_structure=sample,
            padding_width=padding_width,
            overlap_width=overlap_width,
        )
        #sample = padding_mask
    else:
        padding_mask, padding_function = mask_generating_helper_periodic_padding(
            base_shape=sample.shape,
            padding_width=padding_width,
        )
        sample = padding_function(sample)
        padding_mask = BM2SM(padding_mask.to(polymicros.device), sample.to(polymicros.device))

    if True:
        # creating the conditioning function
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
            cond_fn = secondstage_cond_fn,
            images=sample
        )
    
    sample = sample + sample_mean

    # extraction
    sample = final_micro_extractor(sample)

    print(sample.shape)
    sample = np.moveaxis(sample[0].cpu().numpy(), 0, -1)
    print(sample.shape)

    np.save('./figs/TG/periodic_microstructure_example_secondstage.npy', sample)

    f, axes = plt.subplots(1, 3, figsize=[15, 5])

    axes[0].imshow(np.swapaxes(sample[sample.shape[0] // 2, :, :], 0, 1))
    axes[1].imshow(np.swapaxes(sample[:, sample.shape[1] // 2, :], 0, 1))
    axes[2].imshow(np.swapaxes(sample[:, :, sample.shape[2] // 2], 0, 1))

    for ax in axes:
        ax.invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/temp.png', dpi=300)

def visualize_segments():
    '''
    vis
    '''
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    shift = 0
    #sample = torch.roll(torch.from_numpy(sample), shifts=[shift for _ in range(3)], dims=(0, 1, 2)).numpy()
    sample = torch.roll(torch.from_numpy(sample), shifts=[5, ], dims=(2, )).numpy()
    print(sample.shape)

    f, axes = plt.subplots(1, 3, figsize=[15, 5])

    axes[0].imshow(np.swapaxes(sample[sample.shape[0] // 2, :, :], 0, 1))
    axes[1].imshow(np.swapaxes(sample[:, sample.shape[1] // 2, :], 0, 1))
    axes[2].imshow(np.swapaxes(sample[:, :, sample.shape[2] // 2], 0, 1))

    for ax in axes:
        ax.invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/TG/example_periodic.png', dpi=300)

def visualize_postprocessing():
    '''
    vis
    '''
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    full_cube = np.load('./experiments/cutoutcube.npy')
    rogsh_full_cube = GSH.ROGSH(full_cube).reshape(-1, 3)
    full_cube = full_cube.reshape(-1, 3)


    example_structure = sample[:, :, sample.shape[2] // 2]
    example_structure_eulers = np.zeros_like(example_structure)

    # direct projection back
    print(example_structure.shape)
    import itertools

    print('beginning iteration')
    for m, n in itertools.product(range(example_structure.shape[0]), range(example_structure.shape[1])):
        eulers = example_structure[m, n]
        diff_index = np.linalg.norm(rogsh_full_cube - eulers, axis=-1).argmin()
        example_structure_eulers[m, n] = full_cube[diff_index]

    # convert back from GSH
    from skimage.filters import median
    #example_structure, _ = GSH.CHullProj(example_structure)
    #example_structure = np.concatenate([median(example_structure[..., n])[..., None] for n in range(3)], axis=-1)
    #example_structure_eulers = GSH.Rogsh2Euler(example_structure)

    f, axes = plt.subplots(1, 3, figsize=[15, 5])

    axes[0].imshow(np.swapaxes(example_structure, 0, 1))
    axes[1].imshow(np.swapaxes(example_structure_eulers, 0, 1))
    axes[2].plot(np.arange(example_structure.shape[1]), example_structure[example_structure.shape[0] // 2, :, 0], 'k-')

    for ax in axes:
        ax.invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/TG/example_periodic_postprocess.png', dpi=300)

def segment_postprocessing_v2():
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    example_structure = example_structure_eulers = sample[:, :, sample.shape[2] // 2].astype(np.float32)

    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    from PIL import Image
    from skimage.filters import median
    from scipy import ndimage, stats

    def generated_filter(arr, tolerance = 3.0):
        center_pixel = arr.reshape(3, 3)[1, 1]
        if np.isclose(center_pixel, arr, atol=tolerance).sum() < 3.5:
            return stats.mode(arr)[0]
            #return np.median(arr)
        else:
            return center_pixel

    def mode_filter(image, width=3, tolerance=1.0):
        footprint = np.ones((width, width), dtype=bool) # 3x3 neighborhood
        lowlevel_filter = partial(generated_filter, tolerance=tolerance)
        return ndimage.generic_filter(
            image,
            #lambda x: stats.mode(x, keepdims=True)[0][0], # Or stats.mode(x, axis=None).mode[0] for SciPy >=1.11
            lowlevel_filter,
            footprint=footprint,
            mode='wrap',
            #axes=(0, 1),
        )
    
            

    image = (example_structure - example_structure.min()) / (example_structure.max() / example_structure.min()) * 255
    image = image.astype(np.uint8)

    from skimage.morphology import flood, flood_fill
    from skimage.color import rgb2gray, gray2rgb
    import itertools

    def periodic_flood(image, seed_point, tolerance=0.01):
        # 1. Create a larger tiled image (3x3 grid of the original)
        height, width = image.shape
        padded_image_3x3 = np.tile(image, (3, 3))

        # 2. Adjust seed point to the central tile of the 3x3 grid
        seed_point_padded = (seed_point[0] + height, seed_point[1] + width)

        # 3. Run flood_fill on the padded image
        #    The `connectivity` argument might be relevant here.
        #    The mask returned by flood_fill will be the size of padded_image_3x3
        filled_mask_padded = flood(
            padded_image_3x3, 
            seed_point_padded, 
            connectivity=1, 
            tolerance=tolerance
        )

        # filled_image_on_padded = padded_image_3x3.copy() # Make a copy to fill
        # filled_image_on_padded[filled_mask_padded] = fill_value # Apply fill value to a copy

        # 4. Extract the result from the central tile
        # The filled mask for the original image is the central part of filled_mask_padded
        height_slices = [slice(height*n, height*(n+1)) for n in range(3)]
        width_slices = [slice(width*n, width*(n+1)) for n in range(3)]

        results = [filled_mask_padded[hei_slice, wid_slice, None] for hei_slice, wid_slice in itertools.product(height_slices, width_slices)]
        results = np.concatenate(results, axis=-1).max(axis=-1)

        return results

    image = rgb2gray(image)

    observed = np.zeros_like(image)
    new_image = np.zeros_like(image)

    tolerance = 0.020
    min_volume = 5

    grain_index = 1

    for x, y in itertools.product(*[range(dim) for dim in image.shape]):

        if observed[x, y] == 0:
            mask = periodic_flood(image, seed_point=(x, y), tolerance=tolerance)
        
            if mask.sum() < min_volume:
                print('size too small')
                new_image += mask * 0

            else:
                new_image += mask * min(max(0, grain_index), 10)
                grain_index += 3
            
            observed += mask
            if grain_index == 3:
                break
            print('found one!')


    # pack image
    #image = mode_filter(image, tolerance=0.007)
    #image = gray2rgb(image)
    #print(image.min(), image.max())
    #exit()
    #image = image.astype(np.uint8)
    #image = np.concatenate([filter_(image[..., n], tolerance=tolerances[n])[..., None] for n in range(3)], axis=-1)
    #image = np.concatenate([filter_(image[..., n], tolerance=tolerances[n])[..., None] for n in range(3)], axis=-1)

    f, axes = plt.subplots(1, 4, figsize=[20, 5])

    axes[0].imshow(np.swapaxes(example_structure, 0, 1))
    axes[1].imshow(np.swapaxes(image, 0, 1))
    axes[2].imshow(np.swapaxes(new_image, 0, 1))
    axes[3].plot(np.arange(example_structure.shape[1]), example_structure[example_structure.shape[0] // 2, :, 0], 'k-')

    for ax in axes:
        ax.invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/TG/example_periodic_segmentation_with_SAM.png', dpi=300)


def segment_postprocessing():
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    example_structure = example_structure_eulers = sample[:, :, sample.shape[2] // 2].astype(np.float32)

    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    from PIL import Image
    from skimage.filters import median
    from scipy import ndimage, stats

    def generated_filter(arr, tolerance = 3.0):
        center_pixel = arr.reshape(3, 3)[1, 1]
        if np.isclose(center_pixel, arr, atol=tolerance).sum() < 3.5:
            return stats.mode(arr)[0]
            #return np.median(arr)
        else:
            return center_pixel

    def mode_filter(image, width=3, tolerance=1.0):
        footprint = np.ones((width, width), dtype=bool) # 3x3 neighborhood
        lowlevel_filter = partial(generated_filter, tolerance=tolerance)
        return ndimage.generic_filter(
            image,
            #lambda x: stats.mode(x, keepdims=True)[0][0], # Or stats.mode(x, axis=None).mode[0] for SciPy >=1.11
            lowlevel_filter,
            footprint=footprint,
            mode='wrap',
            #axes=(0, 1),
        )
    
    filter_ = mode_filter
    #filter_ = median
    tolerances = [3.0, 3.0, 3.0]

    image = (example_structure - example_structure.min()) / (example_structure.max() / example_structure.min()) * 255
    image = image.astype(np.uint8)
    # pack image
    #from skimage.color import rgb2gray, gray2rgb
    #image = rgb2gray(image)
    #image = mode_filter(image, tolerance=0.007)
    #image = gray2rgb(image)
    #print(image.min(), image.max())
    #exit()
    #image = image.astype(np.uint8)
    #image = np.concatenate([filter_(image[..., n], tolerance=tolerances[n])[..., None] for n in range(3)], axis=-1)
    #image = np.concatenate([filter_(image[..., n], tolerance=tolerances[n])[..., None] for n in range(3)], axis=-1)

    sam = sam_model_registry["default"](checkpoint="../DMN/Data/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    new_image = np.zeros_like(image[..., 0])
    for n, mask in enumerate(masks):
        new_image = new_image + mask['segmentation'] * (n + 1)

    f, axes = plt.subplots(1, 4, figsize=[20, 5])

    axes[0].imshow(np.swapaxes(example_structure, 0, 1))
    axes[1].imshow(np.swapaxes(image, 0, 1))
    axes[2].imshow(np.swapaxes(new_image, 0, 1))
    axes[3].plot(np.arange(example_structure.shape[1]), example_structure[example_structure.shape[0] // 2, :, 0], 'k-')

    for ax in axes:
        ax.invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/TG/example_periodic_segmentation_with_SAM.png', dpi=300)

def project_rogsh_to_euler(
        micro,
        mode = 'orix',
    ):
    from orix.sampling import get_sample_fundamental
    from orix.quaternion import symmetry
    from scipy.spatial import KDTree

    # generate Euler angle list
    if mode.lower() == 'orix':
        cubic_symmetry = symmetry.O
        resolution_degrees = 2
        sampled_orientations = get_sample_fundamental(
            point_group=cubic_symmetry,
            resolution=resolution_degrees,
            method="cubochoric"
        ).to_euler()
        print(sampled_orientations.shape)

        sampled_orientations_rogsh = GSH.ROGSH(sampled_orientations)

        # generate KDtree
        kdtree = KDTree(sampled_orientations_rogsh)
        example_structure_eulers = kdtree.query(micro, k=1)[1]
        example_structure_eulers = sampled_orientations[example_structure_eulers]
    
    elif mode.lower() == 'grid':
        # a grid search
        density = 30
        sampled_orientations = np.meshgrid(
            np.linspace(0, np.pi / 2, density),
            np.linspace(0, np.pi / 2, density),
            np.linspace(0, np.pi / 2, density),
        )
        sampled_orientations = np.concatenate([field.flatten()[..., None] for field in sampled_orientations], axis=-1)
        print(sampled_orientations.shape)

        # convert to ROGSH
        sampled_orientations_rogsh = GSH.ROGSH(sampled_orientations.copy())

        # generate KDtree
        kdtree = KDTree(sampled_orientations_rogsh)
        example_structure_eulers = kdtree.query(micro, k=1)[1]
        example_structure_eulers = sampled_orientations[example_structure_eulers]

    elif mode.lower() == 'michael':
        example_structure_eulers = GSH.Rogsh2Euler(micro)
    
    return example_structure_eulers


def rogsh_grouping():
    ''' grouping '''
    from sklearn import cluster
    from sklearn.cluster import HDBSCAN

    # get base image
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    mode = 'orix'
    example_structure = sample[:, :, sample.shape[2] // 2]

    clusterer = HDBSCAN(
        cluster_selection_epsilon=0.020,
        min_cluster_size=8,
        #p=1,
    )

    example_structure = example_structure.reshape(-1, 3)
    labels = clusterer.fit_predict(example_structure)

    temp = example_structure.copy()

    centers = []
    for label in np.unique(labels):
        if label != -1:
            center = temp[labels == label, :].mean(axis=0)
            temp[labels == label, :] = center
            centers.append(center)
    centers = np.array(centers)
    
    for n, label in enumerate(labels):
        if label == -1:
            vec = example_structure[n]
            min_index = np.linalg.norm(centers - vec, axis=-1).argmin()
            temp[n, :] = centers[min_index, :]
    
    f, ax = plt.subplots(1, 7, figsize=[40, 5], width_ratios=[1.0, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3])
    ax[0].plot(example_structure[:, 0], example_structure[:, 1], 'k.')
    ax[0].plot(temp[:, 0], temp[:, 1], 'rx')
    ax[1].scatter(example_structure[:, 0], example_structure[:, 1], c=labels)
    ax[2].plot(temp[:, 0], temp[:, 1], 'k.')

    # visualizing shapes
    example_structure = example_structure.reshape(82, 82, 3)
    temp = temp.reshape(82, 82, 3)

    mode = 'orix'
    example_structure_euler = project_rogsh_to_euler(example_structure, mode=mode)
    temp_euler = project_rogsh_to_euler(temp, mode=mode)
    temp_euler_rogsh = GSH.ROGSH(temp_euler)

    ax[3].imshow(np.swapaxes(example_structure, 0, 1))
    ax[4].imshow(np.swapaxes(example_structure_euler, 0, 1))

    ax[5].imshow(np.swapaxes(temp, 0, 1))
    ax[6].imshow(np.swapaxes(temp_euler_rogsh, 0, 1))

    ax[0].set_title('Original ROGSH + Centers')
    ax[1].set_title('Original ROGSH - Cluster Label')
    ax[2].set_title('ROGSH Cluster Centers')

    ax[3].set_title('Original Image in ROGSH Space')
    ax[4].set_title('Original Image in Euler Space')

    ax[5].set_title('Clustered Image in ROGSH Space')
    ax[6].set_title('Clustered Image in Euler Space')

    for axi in ax[:-4]:
        axi.set_xlim([-0.41, 0.49])
        axi.set_ylim([-0.48, 0.82])
    
    ax[3].invert_yaxis()
    ax[4].invert_yaxis()
    ax[5].invert_yaxis()
    ax[6].invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/TG/example_periodic_postprocess.png', dpi=300)
    exit()


def rogsh_projection():
    from orix.sampling import get_sample_fundamental
    from orix.quaternion import symmetry
    from scipy.spatial import KDTree

    # get base image
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    mode = 'grid'
    example_structure = sample[:, :, sample.shape[2] // 2]
    print(example_structure.shape)

    #example_structure, _ = GSH.CHullProj(example_structure)
    #temp = example_structure.reshape(-1, 3)
    #f, ax = plt.subplots(1, 1, figsize=[5, 5])
    #ax.plot(temp[:, 0], temp[:, 1], 'k.')
    #f.tight_layout()
    #f.savefig('./figs/TG/example_periodic_postprocess.png', dpi=300)
    #exit()

    #example_structure = np.concatenate([median(example_structure[..., n])[..., None] for n in range(3)], axis=-1)
    #example_structure_eulers = GSH.Rogsh2Euler(example_structure)

    # generate Euler angle list
    if mode.lower() == 'orix':
        cubic_symmetry = symmetry.O
        resolution_degrees = 3
        sampled_orientations = get_sample_fundamental(
            point_group=cubic_symmetry,
            resolution=resolution_degrees,
            method="cubochoric"
        ).to_euler()
        print(sampled_orientations.shape)

        sampled_orientations_rogsh = GSH.ROGSH(sampled_orientations)

        # generate KDtree
        kdtree = KDTree(sampled_orientations_rogsh)
        example_structure_eulers = kdtree.query(example_structure, k=1)[1]
        example_structure_eulers = sampled_orientations[example_structure_eulers]
    
    elif mode.lower() == 'grid':
        # a grid search
        density = 30
        sampled_orientations = np.meshgrid(
            np.linspace(0, np.pi / 2, density),
            np.linspace(0, np.pi / 2, density),
            np.linspace(0, np.pi / 2, density),
        )
        sampled_orientations = np.concatenate([field.flatten()[..., None] for field in sampled_orientations], axis=-1)
        print(sampled_orientations.shape)

        # convert to ROGSH
        sampled_orientations_rogsh = GSH.ROGSH(sampled_orientations.copy())

        # generate KDtree
        kdtree = KDTree(sampled_orientations_rogsh)
        example_structure_eulers = kdtree.query(example_structure, k=1)[1]
        example_structure_eulers = sampled_orientations[example_structure_eulers]

    elif mode.lower() == 'michael':
        example_structure_eulers = GSH.Rogsh2Euler(example_structure)

    # generate plotting
    f, axes = plt.subplots(1, 3, figsize=[15, 5])

    axes[0].imshow(np.swapaxes(example_structure, 0, 1))
    axes[1].imshow(np.swapaxes(example_structure_eulers, 0, 1))
    axes[2].plot(np.arange(example_structure.shape[1]), example_structure[example_structure.shape[0] // 2, :, 0], 'k-')

    for ax in axes:
        ax.invert_yaxis()

    f.tight_layout()
    f.savefig('./figs/TG/example_periodic_postprocess.png', dpi=300)

    exit()


if __name__ == '__main__':
    #main()
    #visualize_postprocessing()
    #segment_postprocessing_v2()
    #rogsh_projection()
    rogsh_grouping()
    #visualize_segments()
    #visualize_data()