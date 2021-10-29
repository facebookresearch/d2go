from io import BytesIO

import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.utils.file_io import PathManager
from PIL import Image


def read_image_with_prefetch(file_name, format=None, prefetched=None):
    if prefetched is None:
        return utils.read_image(file_name, format)

    image = Image.open(BytesIO(prefetched.numpy().view()))
    # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
    image = utils._apply_exif_orientation(image)
    return utils.convert_PIL_to_numpy(image, format)


def read_sem_seg_file_with_prefetch(file_name: str, prefetched=None):
    """
    Segmentation mask annotations can be stored as:
      .PNG files
      .npy uncompressed numpy files
    """
    assert file_name.endswith(".png") or file_name.endswith(".npy")
    sem_seg_type = file_name[-len(".---") :]
    if sem_seg_type == ".png":
        return read_image_with_prefetch(file_name, format="L", prefetched=prefetched)
    elif sem_seg_type == ".npy":
        if prefetched is None:
            with PathManager.open(file_name, "rb") as f:
                return np.load(f)
        else:
            return prefetched.numpy()
