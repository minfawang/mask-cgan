import base64
import glob
import io
import numpy as np
import os
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
                                   transforms.RandomCrop(IMG_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5]) ])


tensor2image = model_utils.tensor2image


def path2Tensor(path: Text) -> torch.Tensor:
  """Converts a file path to input tensor of generator."""
  return _transforms(Image.open(path))


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
  """Input of this function is the output of generator -- unnormalized batch image tensor."""
  return image2DataUrl(tensor2image(batch_img_tensor))


def image2DataUrl(batch_img_tensor: torch.Tensor) -> Text:
  """Input of this function is normalized batch image tensor."""
  buff = io.BytesIO()
  torchvision.utils.save_image(batch_img_tensor, buff, format='jpeg')
  b64_image = base64.b64encode(buff.getvalue()).decode('utf-8')
  return f'data:image/jpeg;base64,{b64_image}'


if __name__ == '__main__':
  """
  export const DEFAULT_REAL_A_SRC = ['<src1>', '<src2>']
  """
  filepaths = sorted(glob.glob('model/datasets/horse2zebra/train/A/*.*'))
  data_urls = []
  for filepath in filepaths[:10]:
    img_tensor = _transforms(Image.open(filepath))
    batch_img_tensor = torch.unsqueeze(img_tensor, dim=0)
    data_url = tensor2DataUrl(batch_img_tensor)
    data_urls.append(data_url)

  data_urls_str = '",\n"'.join(data_urls)
  output = f'export const DEFAULT_REAL_A_SRCS = [{data_urls_str}]'
  with open('client/default_real_a_srcs.js', 'w') as fout:
    fout.write(output)
