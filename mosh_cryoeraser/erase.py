from pathlib import Path

import numpy as np
import tifffile
import mrcfile
from cryoeraser.erase import erase_2d

data_directory = Path('output')
masks_directory = Path('masks')
output_directory = Path('erased')
output_directory.mkdir(exist_ok=True)

for image_file in data_directory.glob('*.tif'):
    mask_file = masks_directory / image_file.name
    if not mask_file.exists():
        continue
    image = np.array(tifffile.imread(image_file))
    mask = np.array(tifffile.imread(mask_file))
    erased_image = erase_2d(image=image, mask=mask)
    output_filename = output_directory / f'{image_file.stem}.mrc'
    mrcfile.write(output_filename, erased_image.astype(np.float16), overwrite=True)
    print(f'wrote {output_filename}')

