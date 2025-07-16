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
import h5py
from torchvision import transforms

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

def logical_median_filter_algorithm(
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

    This logical extension will update any pixel if any
    of its three channels are deemed needed to be updated.
    The non-logical variant updates each channel independently
    which can lead to some undesirable shifting.

    :param micro: a microstructure of shape [SpaceX, SpaceY, Channels]
    '''
    assert micro.shape[-1] == len(tolerances)

    # --------------------------------------------------------------------
    # First Stage: Identifying Islands
    # --------------------------------------------------------------------

    def generated_filter(arr, tolerance, width):
        center_pixel = arr.reshape(width, width)[width // 2, width // 2]
        if np.isclose(center_pixel, arr, atol=tolerance).sum() < (width ** 2 * 0.34):
            return np.nan
        else:
            return center_pixel

    def mask_filter(image, width=width, tolerance=1.0):
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

    update_mask = np.concatenate([mask_filter(micro[..., n], tolerance=tolerances[n])[..., None] for n in range(len(tolerances))], axis=-1)
    update_mask = np.isnan(update_mask).any(axis=-1)

    # --------------------------------------------------------------------
    # Second Stage -- Removing Values
    # --------------------------------------------------------------------

    def apply_masked_mode_filter(A: np.ndarray, B: np.ndarray, window_size: int) -> np.ndarray:
        """
        Applies a mode filter to selected elements of a 3D NumPy array (NxNx3) based on a
        2D boolean mask (NxN), using periodic boundary conditions.

        Args:
            A (np.ndarray): The input 3D NumPy array of shape (N, N, 3) representing an image.
                            The filter will modify this array in-place.
            B (np.ndarray): The 2D boolean NumPy array of shape (N, N), serving as a mask.
                            Where B[i, j] is True, the mode filter will be applied to A[i, j].
            window_size (int): The size of the square window (e.g., 3 for a 3x3 window).
                            Must be an odd positive integer.

        Returns:
            np.ndarray: The modified array A with the mode filter applied to masked elements.
        """

        # --- Input Validation ---
        if not isinstance(A, np.ndarray) or A.ndim != 3 or A.shape[2] != 3:
            print("Error: Input array 'A' must be a 3D NumPy array of shape (N, N, 3).")
            return A
        if not isinstance(B, np.ndarray) or B.ndim != 2 or B.dtype != bool:
            print("Error: Input array 'B' must be a 2D boolean NumPy array of shape (N, N).")
            return A
        if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
            print("Error: Dimensions of A (first two) and B must match (N, N).")
            return A
        if not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0:
            print("Error: window_size must be a positive odd integer.")
            return A

        N = A.shape[0]
        half_window = window_size // 2

        # Create a copy to store results and avoid modifying A while still reading from it
        # in subsequent iterations of the loop for calculating mode.
        # Although the user requested in-place modification, for filter operations,
        # it's often safer to compute on a copy and then update the original.
        # However, the user explicitly asked to "go to the corresponding element in A and apply a mode_filter".
        # This implies the mode filter for A[i,j] depends on the *original* values around A[i,j]
        # rather than newly filtered values. So, we'll iterate and modify A directly.
        # If the intent was for all pixels to be filtered based on the *original* A,
        # a copy would be needed for `window_slice` extraction.
        # For a direct in-place update as requested, we proceed as follows.

        # Iterate only over the elements where B is True
        rows_to_process, cols_to_process = np.where(B)

        for r, c in zip(rows_to_process, cols_to_process):
            # Determine the row and column indices for the current window,
            # applying periodic boundary conditions.
            row_indices = [(r + dr) % N for dr in range(-half_window, half_window + 1)]
            col_indices = [(c + dc) % N for dc in range(-half_window, half_window + 1)]

            # Extract the window from the array A.
            # np.ix_ is used to correctly select 2D slices from a 3D array.
            window_slice = A[np.ix_(row_indices, col_indices)]

            # Calculate the mode for each of the 3 channels
            # .flatten() is used to convert the window_size x window_size array for each channel
            # into a 1D array, as required by scipy.stats.mode.
            mode_channel_0 = stats.mode(window_slice[:, :, 0].flatten(), keepdims=False)[0]
            mode_channel_1 = stats.mode(window_slice[:, :, 1].flatten(), keepdims=False)[0]
            mode_channel_2 = stats.mode(window_slice[:, :, 2].flatten(), keepdims=False)[0]

            # Assign the calculated modes back to the current element A[r, c]
            A[r, c, 0] = mode_channel_0
            A[r, c, 1] = mode_channel_1
            A[r, c, 2] = mode_channel_2

        return A


    micro = apply_masked_mode_filter(micro.copy(), update_mask, window_size=width)

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

# -----------------------------------------------------
# Periodic padding
# -----------------------------------------------------

def resize_with_periodic_padding(image, output_size, interpolation_mode=transforms.InterpolationMode.NEAREST):
    """
    Resizes an image with periodic boundary conditions.

    Args:
        image (PIL.Image or torch.Tensor): The input image.
        output_size (tuple or int): Desired output size (height, width).
        interpolation_mode (torchvision.transforms.InterpolationMode): Interpolation mode.
    """
    if isinstance(image, torch.Tensor):
        image_tensor = image
    else:
        raise ValueError("Input image must be a torch.Tensor")

    original_height, original_width = image_tensor.shape[-2:]

    # Determine padding amount (adjust based on interpolation kernel)
    # For bilinear, 1 pixel is often enough for simple cases. For bicubic, often 2.
    # It's safer to over-pad slightly if unsure.
    padding_h = 6
    padding_w = 6

    # Pad the image periodically using numpy for simplicity
    # Convert to numpy, pad, then convert back to tensor
    np_image = image_tensor.permute(1, 2, 0).numpy() if image_tensor.dim() == 3 else image_tensor.numpy()
    
    # Pad in each dimension (height, width)
    padded_np_image = np.pad(np_image, 
                             ((padding_h, padding_h), (padding_w, padding_w), (0, 0)) if image_tensor.dim() == 3 else ((padding_h, padding_h), (padding_w, padding_w)), 
                             mode='wrap')
    
    padded_image_tensor = torch.from_numpy(padded_np_image).permute(2, 0, 1).unsqueeze(0) if image_tensor.dim() == 3 else torch.from_numpy(padded_np_image)

    # Calculate the scaling factor for the padded image
    scale_h = output_size[0] / original_height
    scale_w = output_size[1] / original_width

    # Calculate the target size for the padded image after resizing
    target_padded_height = int(round(padded_image_tensor.shape[-2] * scale_h))
    target_padded_width = int(round(padded_image_tensor.shape[-1] * scale_w))
    
    resize_transform = transforms.Resize((target_padded_height, target_padded_width), 
                                         interpolation=interpolation_mode)
    
    resized_padded_image = resize_transform(padded_image_tensor)

    # Calculate crop coordinates
    start_h = int(round(padding_h * scale_h))
    start_w = int(round(padding_w * scale_w))
    end_h = start_h + output_size[0]
    end_w = start_w + output_size[1]
    
    final_image = resized_padded_image[..., start_h:end_h, start_w:end_w]

    return final_image

# -----------------------------------------------------
# Main running code
# -----------------------------------------------------

def main_second_filter_and_delete():
    ''' secondary filter and delete process '''
    # Parameters
    toler = 0.010
    width = 3

    remove_indexes = [
        2, 50, 194, 223, 242, 289, 290, 291, 340, 
        341, 352, 353, 386, 385, 458, 448, 473, 602, 
        673, 761, 977, 1015, 1001, 1032, 1048, 1152, 
        1184, 1257, 1265, 1256, 1328, 1336, 1327, 1320
    ]
    resolutions = [32, 64, 128]
    rez_samplers = [transforms.Resize(rez, interpolation=transforms.InterpolationMode.NEAREST) for rez in resolutions]

    load_from_file = './experiments/microstructure_database_May22.h5'
    save_to_file = './experiments/microstructure_database_June5.h5'

    with h5py.File(load_from_file, 'r') as fil:
        total_structures = len(fil['euler']) - len(remove_indexes)

        # Create the save file
        with h5py.File(save_to_file, 'w') as fil_write:
            dset = fil_write.create_dataset(
                'euler',
                shape = (total_structures, 82, 82, 3),
                chunks = (1, 82, 82, 3),
                dtype=np.float64,
                compression='gzip',
                compression_opts=4,
            )

            rez_datasets = []
            for rez in resolutions:
                rez_datasets.append(
                    (
                        fil_write.create_dataset(
                            f'euler_{rez}',
                            shape = (total_structures, rez, rez, 3),
                            chunks = (1, rez, rez, 3),
                            dtype=np.float64,
                            compression='gzip',
                            compression_opts=4,
                        ), 
                        rez
                    )
                )
            
            # curate remaining structures
            new_micro_index = 0
            for micro_index, micro in enumerate(fil['euler']):
                if micro_index not in remove_indexes:
                    # filter
                    micro = logical_median_filter_algorithm(
                        micro.copy(), width=width, tolerances=[toler for _ in range(3)]
                    )

                    # store
                    dset[new_micro_index, ...] = micro

                    # resample
                    for (rez_dset, rez), rez_resampler in zip(rez_datasets, rez_samplers):
                        # down sample
                        #down_micro = torch.movedim(rez_resampler(torch.movedim(torch.from_numpy(micro), -1, 0)), 0, -1).numpy()
                        down_micro = resize_with_periodic_padding(
                             image = torch.movedim(torch.from_numpy(micro), -1, 0),
                             output_size=[rez, rez],
                             interpolation_mode=transforms.InterpolationMode.NEAREST,
                        )
                        down_micro = torch.movedim(down_micro[0], 0, -1).numpy()

                        # save
                        rez_dset[new_micro_index, ...] = down_micro

                    # increment
                    new_micro_index += 1
                    
                else:
                    print(f'Removed: {micro_index}')


def main():
    sample = np.load('./figs/TG/periodic_microstructure_example.npy')
    micro = sample[:, :, sample.shape[2] // 2]

    # PARAMETERS
    save_location = './experiments/microstructure_database_May22.h5'

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

    # extract structures
    sample = np.load('./experiments/cutoutcube.npy')


    # creating a save location
    total_structures = 60 * 24
    structure_index = 0

    with h5py.File(save_location, 'w') as fil:
        dset = fil.create_dataset(
            'euler',
            shape = (total_structures, 82, 82, 3),
            chunks = (1, 82, 82, 3),
            dtype=np.float64,
            compression='gzip',
            compression_opts=4,
        )

        # performing extraction
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
                dset[structure_index] = micro_slice
                structure_index += 1


if __name__ == "__main__":
    #main()
    main_second_filter_and_delete()

