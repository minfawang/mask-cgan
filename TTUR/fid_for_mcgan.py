"""
Example command:
$ python fid_for_mcgan.py

Some configurable variables:
- RUN_ID
- DATASET
- DIR_PATH
"""

import csv
import fid
import os


RUN_ID = 'mask_horse2zebra_h128_nres=3_simpled'
DATASET = 'horse2zebra'
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_id_to_dirs(side):  # side = 'A' | 'B'
  def _out_dir(mask_id):  # mask_id e.g. 'scale=0.50'
    return os.path.join(DIR_PATH, '../model/output', RUN_ID, side, mask_id)

  def _data_dir(split):  # split = 'train' | 'test'
    return os.path.join(DIR_PATH, '../model/datasets', DATASET, split, side)

  id_to_dirs = {
      'scale=0.50': _out_dir('scale=0.50'),
      'scale=0.80': _out_dir('scale=0.80'),
      'scale=1.00': _out_dir('scale=1.00'),
      'train': _data_dir('train'),
      'test': _data_dir('test'),
  }
  return id_to_dirs


def get_out_path(side):
  return os.path.join(DIR_PATH, '../model/output', RUN_ID, f'fid_{side}.tsv')


def grid_with_id(grid, ids):
  N = len(ids)
  assert len(grid) == N
  out_grid = [[''] * (N + 1) for _ in range(N + 1)]

  # Write ids at top row.
  out_grid[0][1:] = list(ids)

  for i in range(1, N + 1):
    # Write ids at left column.
    out_grid[i][0] = ids[i - 1]
    out_grid[i][1:] = list(grid[i - 1])

  return out_grid


if __name__ == '__main__':
  inception_path = None

  data_ids = [
      'scale=0.50',
      'scale=0.80',
      'scale=1.00',
      'train',
      'test',
  ]
  N = len(data_ids)

  for side in ['A', 'B']:
    # Compute fid for each pair of paths and store in fid_grid.
    id_to_dirs = get_id_to_dirs(side)
    fid_grid = [[0] * N for _ in range(N)]
    for i in range(N):
      for j in range(i + 1, N):
        id1, id2 = data_ids[i], data_ids[j]
        paths = id_to_dirs[id1], id_to_dirs[id2]
        fid_value = fid.calculate_fid_given_paths(
            paths, inception_path, low_profile=False)
        # fid_value = 1.0
        fid_grid[i][j] = fid_value
        fid_grid[j][i] = fid_value

    # Add the data ids to the grid
    fid_grid = grid_with_id(fid_grid, data_ids)
    for row in fid_grid:
      print(row)

    # Write the grid data.
    with open(get_out_path(side), 'w') as fout:
      writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
      writer.writerows(fid_grid)
