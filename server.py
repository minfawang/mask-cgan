"""
Example invocation:
$ env FLASK_APP=server.py flask run

OR:
$ python server.py

Build client:
nwb react build client/App.js client/dist/ --title MaskCycleGAN
"""

import glob
import img_utils
import model.test as test_utils
import model.utils as model_utils
import random
import os

from flask import abort, Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, List, Text


# Types
Nets = test_utils.Nets


def load_all_nets(run_ids: List[Text]) -> Dict[Text, Nets]:
  return {
      run_id: test_utils.create_default_nets(run_id)
      for run_id in run_ids
  }


# Globals.
RUN_IDS = [
    'mask_horse2zebra_h128_nres=3_simpled',
    'mask_monet2photo_h128_nres=3_simpled',
    'mask_vangogh2photo_h128_nres=3_simpled',
    'p80mask_horse2zebra_h128_nres=3_simpled',
    'p80mask_vangogh2photo_h128_nres=3_simpled',
]
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__, static_folder='client/dist')
CORS(app)  # Cross origin resource sharing
PORT = 5000

ALL_NETS = load_all_nets(RUN_IDS)
print('Successfully loaded the nets')


@app.route('/generate', methods=['GET', 'POST'])
def handle_generate():
  run_id = request.json['runId']
  if run_id not in ALL_NETS:
    abort(404)

  nets = ALL_NETS[run_id]

  real_src = request.json['realSrc']
  is_a2b = request.json['isA2B']
  real = img_utils.dataUrl2Tensor(real_src)

  mask_percent = request.json['maskPercent']
  mask_percent = '' if mask_percent < 0 else str(mask_percent)
  mask = model_utils.gen_random_mask(mask_percent, shape=real.shape)

  netG_A2B, netG_B2A = nets
  netG = netG_A2B if is_a2b else netG_B2A
  fake = netG(real, mask=mask)

  return jsonify(
      fakeSrc=img_utils.tensor2DataUrl(fake),
      maskSrc=img_utils.image2DataUrl(mask))


@app.route('/rand_imgs/<string:dataset>/<string:side>', methods=['GET', 'POST'])
def handle_rand_imgs(dataset: Text, side: Text):
  num_imgs = 5
  side = side.upper()
  if side not in ['A', 'B']:
    abort(404)

  img_dir = os.path.join(ROOT_DIR, f'model/datasets/{dataset}/train/{side}')
  img_paths = glob.glob(os.path.join(img_dir, '*.*'))
  img_paths = random.sample(img_paths, num_imgs)

  img_srcs = {
      os.path.basename(path): img_utils.path2DataUrl(path)
      for path in img_paths
  }

  return jsonify(imgSrcs=img_srcs)


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
