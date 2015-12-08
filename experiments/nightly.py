from experiments.parameter_tester import parameter_tester
from experiments.parameter_tester import CONFIG


def ant_init():
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='rectangle')
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='circle')
    parameter_tester('ant_init', ['weighted', 'static', 'on_global_best'], data_set='semicircle')
# parameter_tester('tau_init', [CONFIG['tau_max'], CONFIG['tau_min']])
# parameter_tester('decay_type', ['probabilistic', 'gradual'])


def evap_strategy():
    # parameter_tester('decay_type', ['probabilistic', 'gradual'], data_set='rectangle')
    parameter_tester('decay_type', ['probabilistic', 'gradual'], data_set='circle')
    parameter_tester('decay_type', ['probabilistic', 'gradual'], data_set='semicircle')

ant_init()