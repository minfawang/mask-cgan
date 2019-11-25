import base64
import io
import numpy as np
import re
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from typing import Text
import model.utils as model_utils

# Remember to match with test.transforms.
IMG_SIZE = 128
_transforms = transforms.Compose([ transforms.Resize(IMG_SIZE, Image.BICUBIC),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5]) ])


def dataUrl2NpArr(url: Text) -> np.ndarray:
  """Converts a data_url string to np array.

  Returns:
    np.array of shape (H, W, C). uint8 in range [0, 255].

  Reference:
  https://www.reddit.com/r/learnpython/comments/6lqsrp/converting_a_dataurl_to_numpy_array/
  """
  imgstr = re.search(r'base64,(.*)', url).group(1)
  image_bytes = io.BytesIO(base64.b64decode(imgstr))
  im = Image.open(image_bytes)
  return np.array(im)


def dataUrl2Tensor(url: Text) -> torch.Tensor:
  imgstr = re.search(r'base64,(.*)', url).group(1)
  image_bytes = io.BytesIO(base64.b64decode(imgstr))
  im = Image.open(image_bytes)
  img_tensor = _transforms(im)  # (C, H, W)
  batch_img_tensor = torch.unsqueeze(img_tensor, dim=0)  # (1, C, H, W)
  return batch_img_tensor


def tensor2DataUrl(batch_img_tensor: torch.Tensor) -> Text:
  image = model_utils.tensor2image(batch_img_tensor)
  buff = io.BytesIO()
  torchvision.utils.save_image(image, buff, format='jpeg')
  b64_image = base64.b64encode(buff.getvalue()).decode('utf-8')
  return f'data:image/jpeg;base64,{b64_image}'
