import os
import pickle
import sys
import json
import os.path as osp
from time import strftime
from datetime import datetime

from config import SAVE_DIR


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def clear_current_line():
    sys.stdout.write("\r")
    sys.stdout.flush()


def save_object(obj, parent_folder='', file_name=None):
    if parent_folder != '':
        directory = osp.join(SAVE_DIR, parent_folder)
    else:
        directory = osp.join(SAVE_DIR, strftime("%Y-%m-%d_%H%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
    name = os.path.join(directory, file_name)
    pickle.dump(obj, open(name + ".pickle", "wb"))


def save_dict(dictionary, parent_folder='', file_name=None):
    if parent_folder != '':
        directory = osp.join(SAVE_DIR, parent_folder)
    else:
        directory = osp.join(SAVE_DIR, strftime("%Y-%m-%d_%H%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
    name = os.path.join(directory, file_name)
    with open(name, "w") as json_file:
        json_file.write(json.dumps(dictionary, sort_keys=True, indent=2, separators=(',', ': ')))


def save_string_to_file(string, parent_folder=None, file_name=None):
    if parent_folder is None:
        directory = osp.join(SAVE_DIR, strftime("%Y-%m-%d_%H%M"))
    else:
        directory = osp.join(SAVE_DIR, parent_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
    name = os.path.join(directory, file_name)
    with open(name, "w") as f:
        f.write(string)


def normalize(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)

    normalize_const = 1.0 / values.sum()
    return values * normalize_const
