from typing import Tuple

import numpy as np
import einops

from cryoeraser.image_statistics import estimate_local_mean, estimate_standard_deviation


def erase_2d(
    image: np.ndarray,
    mask: np.ndarray,
    background_model_resolution: Tuple[int, int] = (5, 5),
    n_background_samples: int = 20000,
) -> np.ndarray:
    """Erase image features in masked regions.

    Parameters
    ----------
    image: np.ndarray
        `(..., h, w)` array of 2D images.
    mask: np.ndarray
        `(..., h, w)` array of masks defining regions to be erased.
    background_model_resolution: tuple[int, int]
        The spatial resolution `(nh, nw)` of the background intensity model.
    n_background_samples: int
        The number of samples used to estimate the background intensity.

    Returns
    -------
    erased: np.ndarray
        `(..., h, w)` array of 2D images with masked regions erased.
    """
    image, ps = einops.pack([image], pattern='* h w')
    mask, _ = einops.pack([mask], pattern='* h w')
    erased = np.stack(
        [
            _erase_single_image(
                image=_image,
                mask=_mask,
                background_model_resolution=background_model_resolution,
                n_background_samples=n_background_samples,
            )
            for _image, _mask
            in zip(image, mask)
        ], axis=0)
    [erased] = einops.unpack(erased, pattern='* h w', packed_shapes=ps)
    return erased


def _erase_single_image(
    image: np.ndarray,
    mask: np.ndarray,
    background_model_resolution: Tuple[int, int] = (5, 5),
    n_background_samples: int = 20000,
) -> np.ndarray:
    """Replace regions of an image with gaussian noise matching local image statistics.


    Parameters
    ----------
    image: np.ndarray
        `(h, w)` array containing image data for erase.
    mask: np.ndarray
        `(h, w)` binary mask separating foreground from background pixels.
        Foreground pixels (value == 1) will be inpainted.
    background_model_resolution: tuple[int, int]
        Number of points in each image dimension for the background mean model.
        Minimum of two points in each dimension.
    n_background_samples: int
        Number of sampling points for background mean estimation.

    Returns
    -------
    inpainted_image: torch.Tensor
        `(h, w)` array containing image data inpainted in the foreground pixels of the mask
        with gaussian noise matching the local mean and global standard deviation of the image
        for background pixels.
    """
    image = image.astype(np.float64)
    inpainted_image = np.copy(image)
    mask = mask.astype(bool)
    local_mean = estimate_local_mean(
        image=image,
        mask=np.logical_not(mask),
        resolution=background_model_resolution,
        n_samples=n_background_samples,
    )

    # fill foreground pixels with local mean
    idx_foreground = np.argwhere(mask == True)
    idx_foreground = (idx_foreground[:, 0], idx_foreground[:, 1])
    inpainted_image[idx_foreground] = local_mean[idx_foreground]

    # add noise with mean=0 std=background std estimate
    n_pixels_to_inpaint = len(idx_foreground[0])
    background_std = estimate_standard_deviation(
        image=image, mask=np.logical_not(mask)
    )
    noise = np.random.normal(loc=0, scale=background_std, size=n_pixels_to_inpaint)
    inpainted_image[idx_foreground] += noise
    return inpainted_image
