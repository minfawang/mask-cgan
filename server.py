"""
Example invocation:
$ env FLASK_APP=server.py flask run

OR:
$ python server.py
"""

from flask import Flask, request, jsonify
import os
import torch
import model.test as test_utils

# TODO: Load model. Set up a server to handle requests with custom inputs.
# See the following URL for an example implementation:
# https://medium.com/datadriveninvestor/deploy-your-pytorch-model-to-production-f69460192217
class DummyOption(object):
    """Dummy Option object, created to just hold the option attributes."""
    pass


def get_default_option():
    opt = DummyOption()
    opt.run_id = 'mask_horse2zebra_h128_nres=3_simpled'
    opt.max_n = 1
    opt.root_dir = os.path.join(os.path.abspath('.'), 'model')
    opt = test_utils.update_opt(opt)
    return opt


def create_default_nets():
    """Creates the net with default options."""
    opt = get_default_option()
    nets = test_utils.create_nets(opt)
    return nets


# Globals.
app = Flask(__name__)
PORT = 5000
nets = create_default_nets()
print('Successfully loaded the nets')


@app.route('/')
def hello():
  # name = request.args.get('name', 'World')

  # TODO: update mask and real_A values.
  netG_A2B, netG_B2A = nets
  mask = torch.ones(1, 3, 128, 128)
  real_A = torch.zeros(1, 3, 128, 128)
  fake_B = netG_A2B(real_A, mask=mask)

  return jsonify(fake_B=list(fake_B.shape))
  # return f'Hello, {escape(name)}!'


if __name__ == '__main__':
  # 0.0.0.0 will be externally available.
  # 127.0.0.1 will be internally available.
  app.run(host='127.0.0.1', debug=True, port=PORT)
