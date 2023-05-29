import numpy as np
from cryoeraser import erase_2d


def test_erase_single_image():
    image = np.zeros((28, 28))
    image[12:16, 12:16] = 1
    mask = image.copy()
    erased = erase_2d(image=image, mask=mask)
    assert erased.shape == image.shape
    assert np.allclose(erased, np.zeros((28, 28)))


def test_erase_multiple():
    image = np.zeros((2, 28, 28))
    image[:, 12:16, 12:16] = 1
    mask = image.copy()
    erased = erase_2d(image=image, mask=mask)
    assert erased.shape == image.shape
    assert np.allclose(erased, np.zeros((2, 28, 28)))
