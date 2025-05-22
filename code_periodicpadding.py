'''
A complete end to end implementation of the periodic padding
and segmentation pipeline.
'''
import torch

from Unet import UnetND as UnetND_Nonperiodic
from UnetP import UnetND as UnetND_periodic

from CUBEPlot import plot_cube, plot_grid
from ROGSHUtils import ROGSH432
GSH = ROGSH432()

from functools import partial

from MOSMKernel import MultiouputSpectralMixtureKernel, LHSParams
from GRFSampler import TwoPCorrelation
import typing
import numpy as np

from scipy import ndimage, stats
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

from orix.sampling import get_sample_fundamental
from orix.quaternion import symmetry
from scipy.spatial import KDTree

import itertools, json, argparse
from diffusion import Diffusion

# -----------------------------------------------------
# Final postprocessing projector algorithms
# -----------------------------------------------------

class OrixGSH_Projector(object):
    '''
    This is a class that uses Orix to project
    ROGSH coefficients back into euler 
    angles that are inside of the fundamental zone.
    '''
    def __init__(self, resolution_degrees: float=2.0):
        # sample fundamental zone euler angles
        cubic_symmetry = symmetry.O
        self.sampled_orientations = get_sample_fundamental(
            point_group=cubic_symmetry,
            resolution=resolution_degrees,
            method="cubochoric"
        ).to_euler()

        # convert to equivalent ROGSH coefficients
        self.sampled_orientations_rogsh = GSH.ROGSH(self.sampled_orientations)

        # generate KDTree
        self.rogsh_kdtree = KDTree(self.sampled_orientations_rogsh)
    
    def Rogsh2Euler(self, rogsh):
        '''
        :param rogsh: a array of rogsh values [..., 3]
        '''
        assert rogsh.shape[-1] == 3
        eulers_indexes = self.rogsh_kdtree.query(rogsh, k=1)[1]
        return self.sampled_orientations[eulers_indexes]

    def Euler2Rogsh(self, euler):
        '''
        :param euler: a array of bunge convention euler angles [..., 3]
        '''
        assert euler.shape[-1] == 3
        return GSH.ROGSH(euler)

def cluster_cleaning(
        micro: np.ndarray, 
        cluster_selection_epsilon: float=0.020,
        min_cluster_size: int=8,
    ):
    '''
    This uses HDBSCAN clustering to turn noisy grains into 
    individual grains.

    :param micro: a microstructure of shape [SpaceX, SpaceY, Channels]
    '''
    X, Y, C = micro.shape

    clusterer = HDBSCAN(
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_cluster_size=min_cluster_size,
    )

    micro = micro.reshape(-1, 3)
    labels = clusterer.fit_predict(micro)

    # inefficient centering method
    centers = []
    for label in np.unique(labels):
        if label != -1:
            center = micro[labels == label, :].mean(axis=0)
            micro[labels == label, :] = center
            centers.append(center)
    centers = np.array(centers)
    
    # inefficient searching method.
    for n, label in enumerate(labels):
        if label == -1:
            vec = micro[n]
            min_index = np.linalg.norm(centers - vec, axis=-1).argmin()
            micro[n, :] = centers[min_index, :]
    
    # reshaping
    micro = micro.reshape(X, Y, C)
    return micro


def median_filter_algorithm(
        micro: np.ndarray, 
        width: int=3,
        tolerances: list[float]=[3.0, 3.0, 3.0]
    ):
    '''
    This is a custom filtering algorithm that
    replaces a given pixel with the most common
    pixel in its neighborhood if the pixel's value is
    too uncommon in the neighorhood.

    Meant to handle unlikely single pixel islands.

    :param micro: a microstructure of shape [SpaceX, SpaceY, Channels]
    '''
    assert micro.shape[-1] == len(tolerances)

    def generated_filter(arr, tolerance, width):
        center_pixel = arr.reshape(width, width)[width // 2, width // 2]
        if np.isclose(center_pixel, arr, atol=tolerance).sum() < (width ** 2 * 0.34):
            return stats.mode(arr)[0]
            #return np.median(arr)
        else:
            return center_pixel

    def mode_filter(image, width=width, tolerance=1.0):
        footprint = np.ones((width, width), dtype=bool) # 3x3 neighborhood
        lowlevel_filter = partial(generated_filter, tolerance=tolerance, width=width)
        return ndimage.generic_filter(
            image,
            #lambda x: stats.mode(x, keepdims=True)[0][0], # Or stats.mode(x, axis=None).mode[0] for SciPy >=1.11
            lowlevel_filter,
            footprint=footprint,
            mode='wrap',
            #axes=(0, 1),
        )

    micro = np.concatenate([mode_filter(micro[..., n], tolerance=tolerances[n])[..., None] for n in range(len(tolerances))], axis=-1)
    return micro

# -----------------------------------------------------
# PolyMicros Projecting
# -----------------------------------------------------

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

class PolyMicrosPeriodicPadder(object):
    '''
    This is an object that facilitates periodic padding 
    using the PolyMicros foundation model.
    '''
    def __init__(self, 
            config_dir = "./model_checkpoints/config.json",
            model_dir = "./model_checkpoints/PolyMicros.ckpt",
            def_size = 64,
            padding_width = 18,
            overlap_width = 10,
        ):

        # loading the diffusion model
        self.polymicros = load_polymicros(
            model_ckpt = model_dir,
            config = config_dir,
            periodic=False,
            default_spatial_voxelization=def_size + padding_width + overlap_width,
        )

        self.padding_width = padding_width
        self.overlap_width = overlap_width


    @torch.no_grad()
    def __call__(self, 
            sample: torch.Tensor,
            steps = 100,
            skip = 0.85,
            numpy_form: bool=True,
        ):
        # finalizing the parameters
        skip = int(skip * steps)

        if numpy_form:
            sample = torch.from_numpy(sample).to(device=self.polymicros.device).float()
            sample = torch.movedim(sample, -1, 0)[None, ...]

        # removing the mean
        sample_mean = sample.mean(dim=(-3, -2, -1), keepdims=True)
        sample = sample - sample_mean

        # creating the padding function
        padding_mask, masked_cond_fn, secondstage_cond_fn, final_micro_extractor = nanmask_generating_helper_periodic_padding(
            original_structure=sample,
            padding_width=self.padding_width,
            overlap_width=self.overlap_width,
        )

        # creating the conditioning function
        partial_masked_cond_fn = partial(
            masked_cond_fn, 
            masks = padding_mask,
        )

        # conditional sampling
        sample = self.polymicros.sample(
            steps=steps,
            cond_fn=partial_masked_cond_fn,
        )

        # final unconditional refinement
        sample = self.polymicros.sample(
            steps=steps,
            skip = skip,
            images=sample
        )

        # returning the mean
        sample = sample + sample_mean

        # extraction
        sample = final_micro_extractor(sample)
        
        # return in numpy form if desired.
        if numpy_form:
            return np.moveaxis(sample[0].cpu().numpy(), 0, -1)
        else:
            return sample

# -----------------------------------------------------
# extraction methods
# -----------------------------------------------------

def extract_3d_volumes(micro, extracted_dim: int=64):
    '''
    extract smaller 3D volumes from a large 3D volumes. 
    I want to balance amount cropped on either size

    Assumes micro is in the form [SpaceX, SpaceY, SpaceZ, 3]
    '''
    num_segments = [dim // extracted_dim for dim in micro.shape[:-1]]
    starting_index = [
        (dim - num_seg * extracted_dim) // 2 \
            for dim, num_seg in zip(micro.shape[:-1], num_segments)
    ]

    for x_segment, y_segment, z_segment in itertools.product(*[range(num_seg) for num_seg in num_segments]):
        yield micro[
            starting_index[0] + x_segment * extracted_dim: starting_index[0] + (x_segment + 1) * extracted_dim,
            starting_index[1] + y_segment * extracted_dim: starting_index[1] + (y_segment + 1) * extracted_dim,
            starting_index[2] + z_segment * extracted_dim: starting_index[2] + (z_segment + 1) * extracted_dim,
            ...,
        ].copy()


def extract_2d_slices_from_3d_volumes(micro, slice_frequency, offset: typing.Optional[int]=0):
    '''
    This method extracts 2D slices from 3D slices. It will go through the 
    domain and extract slice at an offset frequency defined by slice_frequency.

    assumes that the offset is done on both the top and the bottom

    uses numpy convention: micro = [SpaceX, SpaceY, SpaceZ, 3]
    '''
    dimension = [dim - offset * 2 for dim in micro.shape[:-1]]
    num_slices = [dim // slice_frequency for dim in dimension]

    for x_slice in range(num_slices[0]):
        yield micro[offset + x_slice * slice_frequency, :, :, :].copy()
    
    for y_slice in range(num_slices[1]):
        yield micro[:, offset + y_slice * slice_frequency, :, :].copy()
    
    for z_slice in range(num_slices[2]):
        yield micro[:, :, offset + z_slice * slice_frequency, :].copy()

def main():
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    micro = sample[:, :, sample.shape[2] // 2]

    # PARAMETERS

    # Polymicros parameters
    extracted_dim = 64
    steps = 100
    skip = 0.85
    padding_width = 18
    overlap_width = 10
    config_dir = "./model_checkpoints/config.json"
    model_dir = "./model_checkpoints/PolyMicros.ckpt"

    # slicing parameters
    slice_frequency = 10
    offset = 0

    # postprocessing parameters
    toler = 0.010
    width = 5
    resolution_degree = 1.5

    # loading useful objects
    gsh_projector = OrixGSH_Projector(resolution_degrees=resolution_degree)
    polymicros_padder = PolyMicrosPeriodicPadder(
        config_dir=config_dir,
        model_dir=model_dir,
        def_size=extracted_dim,
        padding_width=padding_width,
        overlap_width=overlap_width,
    )

    collected_slices = []

    # extract structures
    sample = np.load('./experiments/cutoutcube.npy')

    for micro in extract_3d_volumes(sample, extracted_dim=extracted_dim):
        # convert to ROGSH
        micro = GSH.ROGSH(micro)

        # pad using polymicros
        padded_micro = polymicros_padder(
            sample = micro,
            steps = steps,
            skip = skip,
            numpy_form = True,
        )

        # extract slices:
        for slice_index, micro_slice in enumerate(extract_2d_slices_from_3d_volumes(
                padded_micro, 
                slice_frequency=slice_frequency,
                offset = offset,
            )):
            # first cluster
            micro_slice = cluster_cleaning(micro_slice.copy())

            # then clean
            micro_slice = median_filter_algorithm(micro_slice.copy(), width=width, tolerances=[toler for _ in range(3)])

            # project back to Euler
            micro_slice = gsh_projector.Rogsh2Euler(micro_slice.copy())

            # append to collection
            collected_slices.append(micro_slice)

            if len(collected_slices) == 18:
                f, axes = plt.subplots(3, 6, figsize=[30, 15])

                for ax, slic in zip(axes.flatten(), collected_slices):
                    ax.imshow(np.swapaxes(slic, 0, 1))
                    ax.invert_yaxis()

                f.tight_layout()
                f.savefig('./figs/TG/example_extracted_structures.png', dpi=300)

                exit()



if __name__ == "__main__":
    main()

