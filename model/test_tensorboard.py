import torch
from torchviz import make_dot
from models import Generator


class Option(object):
  pass


opt = Option()
opt.input_nc = 3  # input number of channels.
opt.output_nc = 3  # output number of channels.

netG_A2B = Generator(opt.input_nc, opt.output_nc)
real_A = torch.zeros(1, 3, 256, 256)
fake_B = netG_A2B(real_A)

graph = make_dot(fake_B, params=dict(netG_A2B.named_parameters()))
graph.render('output/netG_A2B_model.gv')

"""
The block below tried to display the model graph in tensorboard, but the result
is very obscure.

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/test_tensorboard')
writer.add_graph(netG_A2B, real_A)
writer.close()
"""
