"""
Example invocation:
$ env FLASK_APP=server.py flask run

OR:
$ python server.py
"""

import os
import torch
import model.test as test_utils

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import img_utils


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
app = Flask(__name__, static_folder='client/dist')
CORS(app)  # Cross origin resource sharing
PORT = 5000
nets = create_default_nets()
print('Successfully loaded the nets')


@app.route('/generate', methods=['GET', 'POST'])
def handle_generate():
  real_A_src = request.json['realASrc']
  real_A = img_utils.dataUrl2Tensor(real_A_src)

  # TODO: update mask values.
  netG_A2B, netG_B2A = nets
  mask = torch.ones(1, 3, 128, 128)
  fake_B = netG_A2B(real_A, mask=mask)

  return jsonify(fakeBSrc=img_utils.tensor2DataUrl(fake_B))


@app.route('/')
def handle_home():
  # return render_template('index.html')
  return app.send_static_file('index.html')


@app.route('/app<string:filename>')
def handle_app(filename):
  return app.send_static_file(f'app{filename}')


if __name__ == '__main__':
  # 0.0.0.0 will be externally available.
  # 127.0.0.1 will be internally available.
  app.run(host='127.0.0.1', debug=True, port=PORT)
