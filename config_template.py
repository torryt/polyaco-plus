'''
Copy and this file and name the copy 'config.py'
in order to use it as a configuration file.

Change the configurations below freely to suit your needs.
'''
import os
from bunch import Bunch


SAVE_DIR = os.path.expanduser('~') + '/experiments/'


CLASSIFIER_CONFIG = {
    'tau_min':                  0.001,
    'tau_max':                  1.0,
    'tau_init':                 0.001,
    'rho':                      0.02,
    'alpha':                    1,
    'beta':                     0.01,
    'gpu':                      True,
    'multi_level':              True,
    'max_level':                6,
    'level_convergence_rate':   1200,

    'plot':                     False
}
CLASSIFIER_CONFIG = Bunch(**CLASSIFIER_CONFIG)
