'''
Copy and this file and name the copy 'config.py'
in order to use it as a configuration file.

Change the configurations below freely to suit your needs.
'''
import os

SAVE_DIR = os.path.expanduser('~') + '/experiments/'

CLASSIFIER_CONFIG = {
    'run_time':         20,     # Algorithm runtime in seconds
    'tau_min':          0.001,
    'tau_max':          1.0,
    'tau_init':         0.001,
    'rho':              0.02,
    'alpha':            1,
    'beta':             0.01,
    'gpu':              True,
    'granularity':      3,
    'multi_level':      True,
    'max_level':        4,
    'convergence_rate': 800
}