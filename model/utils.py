import datetime
import random
import numpy as np
import sys
import time
import torch
import torchvision

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


def gen_random_mask(scales, shape):
  """Creates a mask of the specific shape.

  Uniformly sample a percent from [0.5, 0.8, 1.0]. There will be p * p of area
  that will be covered by the mask. For example, if in one draw, percent = 0.5,
  then there will be 0.5 * 0.5 = 0.25 percentage of area that will be covered.

  The mask will always be applied in the center.
  """
  assert shape[1] in [1, 3], 'the image shape should have 1 or 3 channels.'
  assert shape[2] == shape[3]
  size = shape[2]
  percent = random.choice([float(scale) for scale in scales.split(',')])
  mid = size // 2

  lo = int(mid - mid * percent)
  hi = int(mid + mid * percent)

  mask = torch.zeros(*shape, requires_grad=False)
  mask[:, :, lo:hi, lo:hi] = 1.0
  return mask


def tensor2image(tensor):
  """Un-normalize a tensor to image.

  Args:
    tensor: shape [batch, C, H, W]. It has been normalized by (x - 0.5) / 0.5

  Returns:
    tensor * 0.5 + 0.5. Should be float in value range [0, 1].
  """
  return tensor * 0.5 + 0.5


class Logger():

  def __init__(self, n_epochs, batches_epoch, log_dir, epoch=0, batch=1):
    self.n_epochs = n_epochs
    self.batches_epoch = batches_epoch
    self.epoch = epoch
    self.batch = batch
    self.prev_time = time.time()
    self.mean_period = 0
    self.writer = SummaryWriter(log_dir=log_dir)

  def step(self):
    ##########################
    # End of epoch
    if self.batch + 1 == self.batches_epoch:
      self.epoch += 1
      self.batch = 0
      sys.stdout.write('\n')
    else:
      self.batch += 1

  def log(self, losses=None, images=None):
    self.mean_period += (time.time() - self.prev_time)
    self.prev_time = time.time()
    global_step = self.epoch * self.batches_epoch + self.batch

    ##########################
    # Log to stdout.
    sys.stdout.write(
        '\rEpoch %03d/%03d [%04d/%04d] -- ' %
        (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

    losses = {name: loss.item() for name, loss in losses.items()}
    loss_str = ' | '.join([
        f'{name}: {loss:.4f}'
        for name, loss in sorted(losses.items())])
    sys.stdout.write(f'{loss_str} -- ')

    for name, loss in losses.items():
      self.writer.add_scalar(name, loss, global_step=global_step)

    batches_done = self.batches_epoch * self.epoch + self.batch + 1
    batches_left = self.batches_epoch * self.n_epochs - batches_done
    sys.stdout.write('ETA: %s' % (datetime.timedelta(
        seconds=batches_left * self.mean_period / batches_done)))

    ##########################
    # Draw images
    for image_name, tensor in images.items():
      grid = torchvision.utils.make_grid(tensor2image(tensor))
      self.writer.add_image(image_name, grid, global_step=global_step)


class ReplayBuffer():

  def __init__(self, max_size=50):
    assert (max_size >
            0), 'Empty buffer or trying to create a black hole. Be careful.'
    self.max_size = max_size
    self.data = []

  def push_and_pop(self, data):
    """Maybe push an element into the buffer and ...

        Args:
            data: List[element].

        Returns:
            Tensor of concatenation of a list of elements.

        If the buffer is not full, then the element in the new data will be
        inserted to the buffer, and will be in the output tensor.
        Otherwise, there will be 50% of the chance that the element will swap
        with an existing element in the buffer, and the existing element will
        be popped.
        """
    to_return = []
    for element in data.data:
      element = torch.unsqueeze(element, 0)
      if len(self.data) < self.max_size:
        self.data.append(element)
        to_return.append(element)
      else:
        if random.uniform(0, 1) > 0.5:
          i = random.randint(0, self.max_size - 1)
          to_return.append(self.data[i].clone())
          self.data[i] = element
        else:
          to_return.append(element)
    return Variable(torch.cat(to_return))


class LambdaLR():

  def __init__(self, n_epochs, offset, decay_start_epoch):
    assert ((n_epochs - decay_start_epoch) >
            0), "Decay must start before the training session ends!"
    self.n_epochs = n_epochs
    self.offset = offset
    self.decay_start_epoch = decay_start_epoch

  def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
        self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm2d') != -1:
    torch.nn.init.normal(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant(m.bias.data, 0.0)
