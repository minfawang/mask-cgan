import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
  """
  out := x + conv_block(x)

  conv_block(x) := pad - conv - ins_norm - ReLU - pad - conv - ins_norm
  """

  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()

    conv_block = [
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_features, in_features, 3),
        nn.InstanceNorm2d(in_features)
    ]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x):
    return x + self.conv_block(x)


class Generator(nn.Module):
  """
  initial := pad - conv - ins_norm - relu
  |
  down_sample := 2 * (conv - ins_norm - relu)
  |
  residual := 9 * residual_blocks
  |
  up_sample := 2 * (conv_trans - ins_norm - relu)
  |
  output := pad - conv - tanh

  timeit:
    - CPU. Batch = 10. Size = 256. n_res = 9.
      - 17.5s +- 260ms.
    - CPU. Batch = 10. Size = 256. n_res = 3.
      - 1.44s +- 65.9ms
    - CPU. Batch = 10. Size = 64. n_res = 3.
      - 696ms +- 38.9ms
    - CPU. Batch = 10. Size = 64. n_res = 9.
      - 1.48s +- 65.1ms
  """

  def __init__(self, input_nc, output_nc, n_residual_blocks=9, use_mask=False):
    super(Generator, self).__init__()

    # Initial convolution block
    model = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_nc, 64, 7),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True)
    ]

    # Downsampling
    in_features = 64
    out_features = in_features * 2
    for _ in range(2):
      model += [
          nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
          nn.InstanceNorm2d(out_features),
          nn.ReLU(inplace=True)
      ]
      in_features = out_features
      out_features = in_features * 2

    # Residual blocks
    for _ in range(n_residual_blocks):
      model += [ResidualBlock(in_features)]

    # Upsampling
    out_features = in_features // 2
    for _ in range(2):
      model += [
          nn.ConvTranspose2d(in_features,
                             out_features,
                             3,
                             stride=2,
                             padding=1,
                             output_padding=1),
          nn.InstanceNorm2d(out_features),
          nn.ReLU(inplace=True)
      ]
      in_features = out_features
      out_features = in_features // 2

    # Output layer
    model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
    self.model = nn.Sequential(*model)

    # TODO: re-evaluate the mask model architecture.
    if use_mask:
      self.mask_model = nn.Conv2d(2 * input_nc, input_nc, 1, stride=1, padding=0)
      self.context_model = nn.Conv2d(2 * input_nc, output_nc, 1, stride=1, padding=0)

  def forward(self, x, mask=None):
    if mask is None:
      return self.model(x)

    # TODO: do we need to multiply mask at the outside?
    x_mask = torch.cat([x * mask, mask], dim=1)  # [batch, 2 * input_nc, H, W]
    x_context = torch.cat([x * (1 - mask), 1 - mask], dim=1)  # [batch, 2 * input_nc, H, W]
    return self.model(self.mask_model(x_mask)) + self.context_model(x_context)


class Discriminator(nn.Module):

  def __init__(self, input_nc, simple_d):
    """Constructor.

    Args:
      input_nc: int. Number of input channels.
      simpld_d: bool. Use simple discriminator or not.

    if simple_d:
      model := 3 * conv - avg_pool
    else:
      model := 5 * conv - avg_pool

    timeit:
      - Running on CPU. Batch of 10. Input size = 256.
        - simple_d: 358ms
        - no simple_d: 1.13s
      - Running on CPU. Batch of 10. Input size = 64.
        - simple_d: 30ms
        - no simple_d: 134ms
    """
    super(Discriminator, self).__init__()

    # A bunch of convolutions one after another
    model = [
        nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    model += [
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    if not simple_d:
      model += [
          nn.Conv2d(128, 256, 4, stride=2, padding=1),
          nn.InstanceNorm2d(256),
          nn.LeakyReLU(0.2, inplace=True)
      ]

      model += [
          nn.Conv2d(256, 512, 4, padding=1),
          nn.InstanceNorm2d(512),
          nn.LeakyReLU(0.2, inplace=True)
      ]

    # FCN classification layer
    last_nc = 128 if simple_d else 512
    model += [nn.Conv2d(last_nc, 1, 4, padding=1)]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    x = self.model(x)
    # Average pooling and flatten
    return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
