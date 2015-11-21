import os
import pickle
import sys
import uuid
import json
from time import strftime
from datetime import datetime

from config import SAVE_DIR


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def save_object(obj, parent_folder='', file_name=None):
    if parent_folder != '':
        directory = os.path.join(SAVE_DIR, parent_folder)
    else:
        directory = os.path.join(SAVE_DIR, strftime("%Y-%m-%d_%H%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
    name = os.path.join(directory, file_name)
    pickle.dump(obj, open(name + ".pickle", "wb"))


def save_dict(dictionary, parent_folder='', file_name=None):
    if parent_folder != '':
        directory = os.path.join(SAVE_DIR, parent_folder)
    else:
        directory = os.path.join(SAVE_DIR, strftime("%Y-%m-%d_%H%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')[:-5]
    name = os.path.join(directory, file_name)
    with open(name, "w") as json_file:
        json_file.write(json.dumps(dictionary, sort_keys=True, indent=2, separators=(',', ': ')))


def normalize(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)

    normalize_const = 1.0 / values.sum()
    return values * normalize_const
