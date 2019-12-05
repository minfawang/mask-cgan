"""
Example invocation:
$ env FLASK_APP=server.py flask run

OR:
$ python server.py

Build client:
nwb react build client/App.js client/dist/ --title MaskCycleGAN
"""

import model.test as test_utils

from flask import abort, Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, List, Text
import img_utils
import model.utils as model_utils


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
]

app = Flask(__name__, static_folder='client/dist')
CORS(app)  # Cross origin resource sharing
PORT = 5000

ALL_NETS = load_all_nets(RUN_IDS)
print('Successfully loaded the nets')


@app.route('/generate/<string:run_id>', methods=['GET', 'POST'])
def handle_generate(run_id):
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
