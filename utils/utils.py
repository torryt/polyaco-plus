import os
import pickle
import sys
import uuid
import json
from time import strftime

from config import SAVE_DIR


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def save_object(object, file_name=None, parent_folder=''):
    directory = SAVE_DIR + strftime("%Y-%m-%d_%H%M") + parent_folder + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = str(uuid.uuid4())
    name = directory + file_name
    pickle.dump(object, open(name + ".pickle", "wb"), 2)


def save_dict(dictionary, file_name=None, parent_folder=''):
    directory = SAVE_DIR + strftime("%Y-%m-%d_%H%M") + parent_folder + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = str(uuid.uuid4())
    name = directory + file_name
    with open(name, "w") as json_file:
        json_file.write(json.dumps(dictionary, sort_keys=True, indent=2, separators=(',', ': ')))
