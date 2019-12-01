#!/usr/bin/python3
r"""
Example invocation:

python train.py \
    --dataroot datasets/monet2photo \
    --run_id=p80mask_monet2photo_h128_nres=3_simpled \
    --size=128 \
    --n_res_blocks=3 \
    --simple_d=1 \
    --use_mask=1 \
    --mask_scales=''

Using cuda:
    --cuda

# To use classic masking scheme
python train.py \
    --run_id=mask_... \
    --use_mask=1 \
    --mask_scales="0.5,0.8,1.0"
"""

import argparse
import itertools
import json
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal, gen_random_mask
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--batch', type=int, default=0, help='starting batch')
parser.add_argument('--run_id', type=str, default='default', help='If existing run_id is found, will resume from that')
parser.add_argument('--simple_d', type=bool, default=False, help='If true, will use simple discriminator that uses 2 conv layers rather than 4.')
parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of ResNet blocks.')
parser.add_argument('--use_mask', type=bool, default=False, help='If true, will apply random mask in each step.')
parser.add_argument('--mask_scales', type=str, default='0.5, 0.8, 1.0', help='Comma separated list of mask percentages')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print(
        "WARNING: You have a CUDA device, "
        "so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc, n_residual_blocks=opt.n_res_blocks, use_mask=opt.use_mask)
netG_B2A = Generator(opt.output_nc, opt.input_nc, n_residual_blocks=opt.n_res_blocks, use_mask=opt.use_mask)
netD_A = Discriminator(opt.input_nc, opt.simple_d)
netD_B = Discriminator(opt.output_nc, opt.simple_d)

netD_Am = Discriminator(opt.input_nc, opt.simple_d)
netD_Bm = Discriminator(opt.output_nc, opt.simple_d)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    netD_Am.cuda()
    netD_Bm.cuda()

# directory that stores all the model checkpoints.
_run_dir = os.path.join('runs', opt.run_id)


def get_state_path(net_name):
    """Example: runs/default/netG_A2B.pth"""
    return os.path.join(_run_dir, f'{net_name}.pth')


def get_log_dir():
    log_dir = os.path.join(_run_dir, 'log/')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def compute_cycle_loss(recovered, real, mask=None):
    if mask is None:
        return criterion_cycle(recovered, real)

    w_cycle_mask = 0.3
    mask_loss = criterion_cycle(recovered * mask, real * mask)
    context_loss = criterion_cycle(recovered * (1 - mask), real * (1 - mask))
    return w_cycle_mask * mask_loss + (1.0 - w_cycle_mask) * context_loss


def get_loss_D(fake, real, netD, fake_buffer):
    # Real loss
    pred_real = netD(real)
    loss_D_real = criterion_GAN(pred_real, target_real)

    # Fake loss
    fake = fake_buffer.push_and_pop(fake)
    pred_fake = netD(fake.detach())
    loss_D_fake = criterion_GAN(pred_fake, target_fake)
    return (loss_D_real + loss_D_fake) * 0.5


# Load model checkpoints
if os.path.exists(_run_dir):
    state_json_path = os.path.join(_run_dir, 'state.json')
    if not os.path.exists(state_json_path):
        print(f'WARNING: Found run directory but missing {state_json_path}. Start from scratch.')
    else:
        # Decode state.json and refreshes the opt.
        with open(state_json_path, 'r') as fin:
            state_json = json.load(fin)
        print('Resuming from epoch {} batch {}'.format(state_json['epoch'],
                                                       state_json['batch']))
        opt.epoch = state_json['epoch']
        opt.batch = state_json['batch']

        netG_A2B.load_state_dict(torch.load(get_state_path('netG_A2B')))
        netG_B2A.load_state_dict(torch.load(get_state_path('netG_B2A')))
        netD_A.load_state_dict(torch.load(get_state_path('netD_A')))
        netD_B.load_state_dict(torch.load(get_state_path('netD_B')))

        if opt.use_mask:
            netD_Am.load_state_dict(torch.load(get_state_path('netD_Am')))
            netD_Bm.load_state_dict(torch.load(get_state_path('netD_Bm')))
else:
    os.makedirs(_run_dir)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    netD_Am.apply(weights_init_normal)
    netD_Bm.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=opt.lr, betas=(0.5, 0.999))

D_A_parameters = itertools.chain(netD_A.parameters(), netD_Am.parameters()) if opt.use_mask else netD_A.parameters()
D_B_parameters = itertools.chain(netD_B.parameters(), netD_Bm.parameters()) if opt.use_mask else netD_B.parameters()
optimizer_D_A = torch.optim.Adam(D_A_parameters, lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B_parameters, lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G,
    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A,
    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B,
    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(
    Tensor(opt.batchSize, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(
    Tensor(opt.batchSize, 1).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_Am_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
fake_Bm_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [
    transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
]
dataloader = DataLoader(
    ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), log_dir=get_log_dir(),
                epoch=opt.epoch, batch=opt.batch)
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        if i < opt.batch:
            continue

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        mask = None
        if opt.use_mask:
            mask = gen_random_mask(opt.mask_scales, real_A.shape)
            if opt.cuda:
                mask = mask.cuda()

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B, mask=mask)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A, mask=mask)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A, mask=mask)
        pred_fake = netD_B(fake_B)
        pred_fake_m = netD_Bm(fake_B * mask) if opt.use_mask else 1.0
        w_gan_mask = 0.7 if opt.use_mask else 0.0
        loss_GAN_A2B = (
            (1 - w_gan_mask) * criterion_GAN(pred_fake, target_real) +
            w_gan_mask * criterion_GAN(pred_fake_m, target_real))

        fake_A = netG_B2A(real_B, mask=mask)
        pred_fake = netD_A(fake_A)
        pred_fake_m = netD_Am(fake_A * mask) if opt.use_mask else 1.0
        loss_GAN_B2A = (
            (1 - w_gan_mask) * criterion_GAN(pred_fake, target_real) +
            w_gan_mask * criterion_GAN(pred_fake_m, target_real))

        # Cycle loss
        recovered_A = netG_B2A(fake_B, mask=mask)
        loss_cycle_ABA = 10.0 * compute_cycle_loss(recovered_A, real_A, mask=mask)

        recovered_B = netG_A2B(fake_A, mask=mask)
        loss_cycle_BAB = 10.0 * compute_cycle_loss(recovered_B, real_B, mask=mask)

        # Total loss
        loss_G = (
            loss_identity_A + loss_identity_B + loss_GAN_A2B +
            loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB)
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Total loss
        loss_D_A_full = (1.0 - w_gan_mask) * get_loss_D(fake_A, real_A, netD_A, fake_A_buffer)
        loss_D_A_mask = (
            w_gan_mask * get_loss_D(fake_A * mask, real_A * mask, netD_Am, fake_Am_buffer)
            if opt.use_mask else 0.0)
        loss_D_A = loss_D_A_full + loss_D_A_mask
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Total loss
        loss_D_B_full = (1.0 - w_gan_mask) * get_loss_D(fake_B, real_B, netD_B, fake_B_buffer)
        loss_D_B_mask = w_gan_mask * (
            get_loss_D(fake_B * mask, real_B * mask, netD_Bm, fake_Bm_buffer)
            if opt.use_mask else 0.0)
        loss_D_B = loss_D_B_full + loss_D_B_mask
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        if i % 100 == 0:
            loss_summary = {
                'loss_G': loss_G,
                'loss_G/identity': (loss_identity_A + loss_identity_B),
                'loss_G/GAN': (loss_GAN_A2B + loss_GAN_B2A),
                'loss_G/cycle': (loss_cycle_ABA + loss_cycle_BAB),
                'loss_D': (loss_D_A + loss_D_B),
                'loss_D/full': (loss_D_A_full + loss_D_B_full),
                'loss_D/mask': (loss_D_A_mask + loss_D_B_mask),
            }
            images_summary = {
                'A/real_A': real_A,
                'A/recovered_A': recovered_A,
                'A/fake_B': fake_B,
                'A/same_A': same_A,

                'B/real_B': real_B,
                'B/recovered_B': recovered_B,
                'B/fake_A': fake_A,
                'B/same_B': same_B,

                'mask': mask,
            }
            logger.log(loss_summary, images=images_summary)
        logger.step()

        if i % 100 == 0:
            # Save models checkpoints
            torch.save(netG_A2B.state_dict(), get_state_path('netG_A2B'))
            torch.save(netG_B2A.state_dict(), get_state_path('netG_B2A'))
            torch.save(netD_A.state_dict(), get_state_path('netD_A'))
            torch.save(netD_B.state_dict(), get_state_path('netD_B'))

            if opt.use_mask:
                torch.save(netD_Am.state_dict(), get_state_path('netD_Am'))
                torch.save(netD_Bm.state_dict(), get_state_path('netD_Bm'))

            with open(os.path.join(_run_dir, 'state.json'), 'w') as fout:
                state_json = {**vars(opt), 'epoch': epoch, 'batch': i}
                json.dump(state_json, fout, indent=2)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
###################################
