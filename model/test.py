#!/usr/bin/python3
r"""
Example command:

python test.py \
    --run_id=horse2zebra_h64 \
    --max_n=100

If you are also interested in getting the image grid, add the following flag:
    --grid
"""

import argparse
import os
from PIL import Image
from tqdm import tqdm
import json

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from models import Generator
from datasets import ImageDataset
from utils import gen_random_mask, tensor2image

parser = argparse.ArgumentParser()

# If update arguments, please also remember to update create_default_net().
parser.add_argument('--run_id', type=str, default='default', help='run id to test on')
parser.add_argument('--max_n', type=int, default=10, help='max number of outputs to generate')
parser.add_argument('--root_dir', type=str, default='.', help='Root directory')
parser.add_argument('--grid', type=bool, default=False, help='If true, output a grid directory')


def update_opt(opt):
    def _load_state_to_opt(opt):
        """Hack: load state json into opt. For definition of args, see train.py"""
        state_json_path = os.path.join(opt.root_dir, f'runs/{opt.run_id}/state.json')
        with open(state_json_path, 'r') as fin:
            state_dict = json.load(fin)
            required_attrs = ['batchSize', 'cuda', 'n_cpu', 'dataroot', 'input_nc', 'output_nc', 'size', 'use_mask']
            for attr in required_attrs:
                assert attr in state_dict, f'missing required attr: {attr}'

            for key, value in state_dict.items():
                setattr(opt, key, value)

    _load_state_to_opt(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not torch.cuda.is_available() and opt.cuda:
        print('WARNING: cuda is not available. Switch to CPU mode')
        opt.cuda = False

    print('\n[NOTE] Currently in testing, --mask_scales are ignored. All outputs use the full mask.\n')
    print(opt)

    return opt


###### Definition of variables ######
# Networks

def create_nets(opt):
    def _get_state_path(net_name):
        """Example: runs/default/netG_A2B.pth"""
        _run_dir = os.path.join(opt.root_dir, 'runs', opt.run_id)
        return os.path.join(_run_dir, f'{net_name}.pth')

    netG_A2B = Generator(opt.input_nc, opt.output_nc, n_residual_blocks=opt.n_res_blocks, use_mask=opt.use_mask)
    netG_B2A = Generator(opt.output_nc, opt.input_nc, n_residual_blocks=opt.n_res_blocks, use_mask=opt.use_mask)

    if opt.cuda:
        print('Convert to cuda \n\n\n\n')

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(_get_state_path('netG_A2B'), map_location='cpu'))
    netG_B2A.load_state_dict(torch.load(_get_state_path('netG_B2A'), map_location='cpu'))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    return netG_A2B, netG_B2A


# Dataset loader
def get_dataloader(opt):
    # Remember to match with img_utils._transforms.
    transforms_ = [ transforms.Resize(opt.size, Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]) ]
    return DataLoader(
        ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


###### Testing######
def gen_images(nets, dataloader, opt):
    def _vbound(divider=4):
        c, h, w = opt.input_nc, opt.size, opt.size
        out_shape = c, int(h // divider), int(w * 10)
        return torch.full(out_shape, fill_value=0.8)

    def _hbound(divider=4):
        """
        Args:
            shape: Either size = 2 for gray image, or 3 for RGB image.
        """
        c, h, w = opt.input_nc, opt.size, opt.size
        out_shape = c, h, int(w // divider)
        return torch.full(out_shape, fill_value=0.8)

    netG_A2B, netG_B2A = nets

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    total = min(len(dataloader), opt.max_n)
    for i, batch in tqdm(enumerate(dataloader), total=total):
        if i >= opt.max_n:
            break

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        if not opt.use_mask:
            # fake_B = 0.5 * (netG_A2B(real_A, mask=mask).data + 1.0)
            # fake_A = 0.5 * (netG_B2A(real_B, mask=mask).data + 1.0)
            fake_B = tensor2image(netG_A2B(real_A, mask=None))
            fake_A = tensor2image(netG_B2A(real_B, mask=None))
            save_image(fake_A, f'output/{opt.run_id}/A/{i+1:04d}.png')
            save_image(fake_B, f'output/{opt.run_id}/B/{i+1:04d}.png')
            continue

        mask_scales = opt.mask_scales.split(',')  # ['0.5', '0.8', '1.0']
        masks = [
            gen_random_mask(size, shape=real_A.shape)
            for size in mask_scales]

        for scale, mask in zip(mask_scales, masks):
            fake_B = tensor2image(netG_A2B(real_A, mask=mask))
            fake_A = tensor2image(netG_B2A(real_B, mask=mask))
            save_image(fake_A, f'output/{opt.run_id}/A/scale={float(scale):.2f}/{i+1:04d}.png')
            save_image(fake_B, f'output/{opt.run_id}/B/scale={float(scale):.2f}/{i+1:04d}.png')

        if not opt.grid:
            continue

        grid = [_vbound()]
        for mask in masks:
            # Generate output
            fake_B = netG_A2B(real_A, mask=mask)
            fake_A = netG_B2A(real_B, mask=mask)
            recovered_A = netG_B2A(fake_B, mask=mask)
            recovered_B = netG_A2B(fake_A, mask=mask)
            same_A = netG_B2A(real_A, mask=mask)
            same_B = netG_A2B(real_B, mask=mask)

            images = [
                _hbound(),
                mask[0],

                # torch.ones_like(real_A[0]),  # separator
                _hbound(),

                tensor2image(real_A[0]),
                tensor2image(fake_B[0]),
                tensor2image(recovered_A[0]),
                tensor2image(same_A[0]),

                _hbound(),

                tensor2image(real_B[0]),
                tensor2image(fake_A[0]),
                tensor2image(recovered_B[0]),
                tensor2image(same_B[0]),

                _hbound(),
            ]

            grid.append(torch.cat(images, dim=2))  # cat on width
            grid.append(_vbound())
        grid = torch.cat(grid, dim=1)  # cat on height

        # Save image files
        save_image(grid, f'output/{opt.run_id}/grid/{i+1:04d}.png')


if __name__ == '__main__':
    opt = parser.parse_args()
    opt = update_opt(opt)

    # Create output dirs if they don't exist
    os.makedirs(f'output/{opt.run_id}/A', exist_ok=True)
    os.makedirs(f'output/{opt.run_id}/B', exist_ok=True)

    if opt.use_mask:
        mask_scales = [float(scale) for scale in opt.mask_scales.split(',')]
        for scale in mask_scales:
            os.makedirs(f'output/{opt.run_id}/A/scale={scale:.2f}', exist_ok=True)
            os.makedirs(f'output/{opt.run_id}/B/scale={scale:.2f}', exist_ok=True)

        if opt.grid:
            os.makedirs(f'output/{opt.run_id}/grid', exist_ok=True)

    nets = create_nets(opt)
    dataloader = get_dataloader(opt)
    gen_images(nets, dataloader, opt)
