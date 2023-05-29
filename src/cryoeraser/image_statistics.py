import warnings
from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import LSQBivariateSpline


def estimate_local_mean(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    resolution: tuple[int, int] = (5, 5),
    n_samples: int = 20000,
):
    """Estimate local mean of an image with a bivariate cubic spline.

    A mask can be provided to define regions which should be used for the
    estimation.

    Parameters
    ----------
    image: np.ndarray
        `(h, w)` array containing image data.
    mask: Optional[np.ndarray]
        `(h, w)` array containing a binary mask specifying foreground
        and background pixels for the estimation.
    resolution: Tuple[int, int]
        Resolution of the local mean estimate in each dimension.
    n_samples: int
        Number of samples taken from foreground pixels for background mean estimation.
        The number of background pixels will be used if this number is greater than the
        number of background pixels.

    Returns
    -------
    local_mean: torch.Tensor
        `(h, w)` array containing a local estimate of the mean intensity.
    """
    input_dtype = image.dtype
    mask = np.ones_like(image, dtype=bool) if mask is None else mask.astype(bool)

    # get a random set of foreground pixels for the estimation
    foreground_sample_idx = np.argwhere(mask == True)
    n_foreground_samples = len(foreground_sample_idx)
    n_samples = min(n_samples, n_foreground_samples)
    selection = np.random.choice(
        foreground_sample_idx.shape[0], size=n_samples, replace=False
    )
    foreground_sample_idx = foreground_sample_idx[selection]
    idx_h, idx_w = foreground_sample_idx[:, 0], foreground_sample_idx[:, 1]
    z = image[idx_h, idx_w]

    # fit a bivariate spline to the data with the specified background model resolution
    ty = np.linspace(0, image.shape[0], num=resolution[0])
    tx = np.linspace(0, image.shape[1], num=resolution[1])
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        background_model = LSQBivariateSpline(idx_h, idx_w, z, tx, ty)

    # evaluate the model over a grid covering the whole image
    idx_h = np.arange(image.shape[-2])
    idx_w = np.arange(image.shape[-1])
    local_mean = background_model(idx_h, idx_w, grid=True)
    return np.asarray(local_mean, dtype=input_dtype)


def estimate_standard_deviation(image: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Estimate the standard deviation of an image from a central crop.

    Parameters
    ----------
    image: np.ndarray
        `(h, w)` image from which stqndard deviation will be estiamted.
    mask: np.ndarray
        `(h, w)` mask of pixels to be used in the estimate.
    """
    mask = np.ones_like(image) if mask is None else mask
    mask = mask.astype(bool)

    h, w = image.shape
    hl, hh = h // 4, (h // 4) * 3
    wl, wh = w // 4, (w // 4) * 3

    image = image[hl:hh, wl:wh]
    mask = mask[hl:hh, wl:wh]
    values = image[mask]
    return float(np.std(values))
