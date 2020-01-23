#!/usr/bin/env python
# Copyright (C) 2018  Mario Juez-Gil <mariojg@ubu.es>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Engine Dataset loading.

Opens h5 file dataset and loads CVV and V datasets.
Loading a subset containing the first N seconds of the dataset
is also possible.

"""

import h5py
import numpy as np
import random

__author__ = "Mario Juez-Gil"
__copyright__ = "Copyright 2018, Mario Juez-Gil"
__credits__ = ["Mario Juez-Gil", "Cesar Garcia-Osorio",
               "Álvar Arnaiz-González", "Carlos López"]
__license__ = "GPLv3"
__version__ = "0.6"
__maintainer__ = "Mario Juez-Gil"
__email__ = "mariojg@ubu.es"
__status__ = "Development"

ALL = "all"  # filter, workload, and frequency shared discriminator

CVV = "cvv"
V = "v"

MIXED_WORKLOAD = "mixed"
MEDIUM_WORKLOAD = "medium"
NO_WORKLOAD = "no"

FREQ_3HZ = "three"
FREQ_12HZ = "twelve"
FREQ_30HZ = "thirty"
FREQ_60HZ = "sixty"
FREQ_LINE = "line"
FREQ_ALL_PAPER = "all_paper"

T_STEPS = 1  # shape index of timesteps value

VA = (CVV, 0)
VB = (CVV, 1)
VC = (CVV, 2)
GND = (CVV, 3)
CA = (CVV, 4)
CB = (CVV, 5)
CC = (CVV, 6)
CN = (CVV, 7)
EC = (CVV, 8)
AREF = (V, 0)
AX = (V, 1)
AY = (V, 2)
AZ = (V, 3)

MASK_ALL = (VA, VB, VC, GND, CA, CB, CC, CN, EC, AREF, AX, AY, AZ)
MASK_NO_GND = (VA, VB, VC, CA, CB, CC, CN, EC, AREF, AX, AY, AZ)
MASK_CVV = (VA, VB, VC, GND, CA, CB, CC, CN, EC)
MASK_CVV_NO_GND = (VA, VB, VC, CA, CB, CC, CN, EC)
MASK_V = (AREF, AX, AY, AZ)

# With this implementation the file is going to be opened each time we request
# a window, which could penalize the performance.
def data_window(size=5, workload=ALL, frequency=ALL, mask=MASK_ALL, norm=True,
                bd=False, root_path="/home/mariojg/research/datasets/motor_faults"):

	def mask_to_dict():
		mask_dict = {
			CVV: [],
			V: []
		}

		for data_filter, col_index in mask:
			mask_dict[data_filter].append(col_index)

		return mask_dict

	dataset_file = f"{root_path}/full_dataset_norm.h5" if norm else f"{root_path}/full_dataset.h5"
	mask = mask_to_dict()

	inputs = []
	outputs = None
	with h5py.File(dataset_file, "r") as ds:
		num_timesteps = {
			CVV: int(ds[f"data/{CVV}"].shape[T_STEPS] * size / 10),
			V: int(ds[f"data/{V}"].shape[T_STEPS] * size / 10)
		}

		ids = None
		if(frequency == "all_paper"):
			ids_three = ds[f"meta/{workload}/three"][()]
			ids_thirty = ds[f"meta/{workload}/thirty"][()]
			ids_line = ds[f"meta/{workload}/line"][()]
			ids = np.concatenate((ids_three, ids_thirty, ids_line))
		else:
			ids = ds[f"meta/{workload}/{frequency}"][()]
		filtered_ids = []
		exps = ds["data/exp"][()]
		for i in ids:
			cond = exps[i][5] == 1 or tuple(exps[i][[2,3,4,5]]) if bd else exps[i][5] == 0
			if(cond):
				filtered_ids.append(i)
		filtered_ids = np.sort(np.array(filtered_ids)).tolist()

		for data_filter, cols in mask.items():
			if len(cols) > 0:
				inputs.append(ds[f"data/{data_filter}"][filtered_ids][:,:num_timesteps[data_filter],cols])

		if len(inputs) == 1:
			inputs = inputs[0]
	return (inputs, outputs)
