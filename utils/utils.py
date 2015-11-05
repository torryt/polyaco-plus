import os
import pickle
import sys
import uuid
from time import strftime, gmtime

from config import SAVE_DIR


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def save_object(object, file_name=None, save_dir=SAVE_DIR):
    directory = save_dir + strftime("%Y-%m-%d_%H%M/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file_name is None:
        file_name = str(uuid.uuid4())
    name = directory + file_name
    pickle.dump(object, open(name + ".pickle", "wb"), 2)
