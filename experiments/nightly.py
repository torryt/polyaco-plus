from experiments.parameter_tester import parameter_tester

CONFIG = {
    'ant_count':    3000,
    'number_runs':  50,
    'tau_min':      0.001,
    'tau_max':      1.0,
    'tau_init':     0.001,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.1,
    'ant_init':     'weighted',
    'decay_type':   'probabilistic',
    'data_set':     'square_space'
}


def beta():
    parameter_tester('beta', [0, 0.01, 0.1, 1.0], config=CONFIG)
beta()


def ant_init():
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'])
# parameter_tester('tau_init', [CONFIG['tau_max'], CONFIG['tau_min']])
# parameter_tester('decay_type', ['probabilistic', 'gradual'])


def evaporation_strategy():
    parameter_tester('decay_type', ['random_type', 'grad_type'])

